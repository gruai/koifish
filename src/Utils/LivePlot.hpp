
/**
 *  SPDX-FileCopyrightText: 2019-2025 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief single-header, zero third-party-code live loss plot.
 *  \author Claude + Yingshi Chen
 */
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <functional>
#include "GST_hashmap.hpp"
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>



#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#ifdef _MSC_VER
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#endif
#else
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <sys/select.h>
#endif

namespace liveplot {

struct Series {
    std::vector<double> x, y;
};
struct Color {
    unsigned char r, g, b;
};
using Pt = std::pair<int, int>;

// State shared between the user's threads and the window thread.
struct Shared {
    std::atomic<bool>  stop{false};
    std::atomic<bool>  dirty{true};
    std::atomic<void*> native{nullptr};   // HWND on Windows (used to wake the window)
    std::atomic<bool>  want_png{false};   // capture a last frame when the window closes
    // Filled by the window thread just before it exits; read by stop() after join().
    int cap_w = 0, cap_h = 0;
    std::vector<unsigned char> cap_rgb;   // cap_w * cap_h * 3, RGB
};

// Minimal PNG writer (RGB8). Uses zlib "stored" blocks, so no compression and no zlib needed.
inline bool write_png(const std::string& path, int w, int h, const std::vector<unsigned char>& rgb) {
    if (w <= 0 || h <= 0 || rgb.size() < (size_t)w * (size_t)h * 3)
        return false;
    static const std::vector<unsigned> tab = [] {
        std::vector<unsigned> t(256);
        for (unsigned n = 0; n < 256; ++n) {
            unsigned c = n;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }();
    auto be32 = [](std::vector<unsigned char>& v, unsigned x) {
        v.push_back((unsigned char)(x >> 24));
        v.push_back((unsigned char)(x >> 16));
        v.push_back((unsigned char)(x >> 8));
        v.push_back((unsigned char)x);
    };
    std::vector<unsigned char> raw;  // scanlines, each prefixed by filter 0
    raw.reserve((size_t)h * (1 + (size_t)w * 3));
    for (int y = 0; y < h; ++y) {
        raw.push_back(0);
        raw.insert(raw.end(), rgb.begin() + (size_t)y * w * 3, rgb.begin() + (size_t)(y + 1) * w * 3);
    }
    std::vector<unsigned char> z{0x78, 0x01};  // zlib header
    for (size_t pos = 0; pos < raw.size();) {
        size_t n = std::min<size_t>(65535, raw.size() - pos);
        z.push_back(pos + n == raw.size() ? 1 : 0);  // BFINAL, BTYPE=00 (stored)
        z.push_back((unsigned char)(n & 0xFF));
        z.push_back((unsigned char)(n >> 8));
        z.push_back((unsigned char)(~n & 0xFF));
        z.push_back((unsigned char)((~n >> 8) & 0xFF));
        z.insert(z.end(), raw.begin() + pos, raw.begin() + pos + n);
        pos += n;
    }
    unsigned a = 1, b = 0;  // Adler-32 of the raw data
    for (unsigned char x : raw) {
        a = (a + x) % 65521u;
        b = (b + a) % 65521u;
    }
    be32(z, (b << 16) | a);

    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
        return false;
    auto chunk = [&](const char* type, const std::vector<unsigned char>& data) {
        std::vector<unsigned char> body(type, type + 4), len, crc;
        body.insert(body.end(), data.begin(), data.end());
        unsigned c = 0xFFFFFFFFu;
        for (unsigned char x : body) c = tab[(c ^ x) & 0xFF] ^ (c >> 8);
        be32(len, (unsigned)data.size());
        be32(crc, c ^ 0xFFFFFFFFu);
        std::fwrite(len.data(), 1, 4, f);
        std::fwrite(body.data(), 1, body.size(), f);
        std::fwrite(crc.data(), 1, 4, f);
    };
    static const unsigned char sig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
    std::fwrite(sig, 1, 8, f);
    std::vector<unsigned char> ihdr;
    be32(ihdr, (unsigned)w);
    be32(ihdr, (unsigned)h);
    ihdr.push_back(8);
    ihdr.push_back(2);
    ihdr.push_back(0);
    ihdr.push_back(0);
    ihdr.push_back(0);
    chunk("IHDR", ihdr);
    chunk("IDAT", z);
    chunk("IEND", {});
    return std::fclose(f) == 0;
}

// ===========================================================================
//  Platform layer: Canvas = tiny 2D drawing surface (rect / line / polyline / text)
// ===========================================================================
#ifdef _WIN32

class Canvas {
   public:
    explicit Canvas(HDC dc) : dc_(dc) {
        SelectObject(dc_, GetStockObject(DEFAULT_GUI_FONT));
        SetBkMode(dc_, TRANSPARENT);
        TEXTMETRICA tm;
        GetTextMetricsA(dc_, &tm);
        th_ = tm.tmHeight;
    }
    int text_h() const { return th_; }
    int text_w(const std::string& s) const {
        SIZE sz{0, 0};
        GetTextExtentPoint32A(dc_, s.c_str(), (int)s.size(), &sz);
        return sz.cx;
    }
    void fill_rect(int x, int y, int w, int h, Color c) {
        RECT r{x, y, x + std::max(w, 0), y + std::max(h, 0)};
        HBRUSH b = CreateSolidBrush(RGB(c.r, c.g, c.b));
        FillRect(dc_, &r, b);
        DeleteObject(b);
    }
    void line(int x1, int y1, int x2, int y2, Color c, int width = 1) { polyline({{x1, y1}, {x2, y2}}, c, width); }
    void polyline(const std::vector<Pt>& p, Color c, int width = 1) {
        if (p.size() < 2)
            return;
        std::vector<POINT> pts;
        pts.reserve(p.size());
        for (auto& q : p) pts.push_back(POINT{q.first, q.second});
        HPEN pen    = CreatePen(PS_SOLID, width, RGB(c.r, c.g, c.b));
        HGDIOBJ old = SelectObject(dc_, pen);
        Polyline(dc_, pts.data(), (int)pts.size());
        SelectObject(dc_, old);
        DeleteObject(pen);
    }
    void text(int x, int y, const std::string& s, Color c) {  // (x,y) = top-left
        SetTextColor(dc_, RGB(c.r, c.g, c.b));
        TextOutA(dc_, x, y, s.c_str(), (int)s.size());
    }

   private:
    HDC dc_;
    int th_ = 14;
};

#else  // ---------------------------- X11 ------------------------------------

class Canvas {
   public:
    Canvas(Display* d, Colormap cm, GC gc, XFontStruct* f) : dpy_(d), cmap_(cm), gc_(gc), font_(f) {}
    void target(Drawable d) { d_ = d; }

    int text_h() const { return font_ ? font_->ascent + font_->descent : 13; }
    int text_w(const std::string& s) const { return font_ ? XTextWidth(font_, s.c_str(), (int)s.size()) : 7 * (int)s.size(); }
    void fill_rect(int x, int y, int w, int h, Color c) {
        fg(c);
        XFillRectangle(dpy_, d_, gc_, x, y, (unsigned)std::max(w, 0), (unsigned)std::max(h, 0));
    }
    void line(int x1, int y1, int x2, int y2, Color c, int width = 1) {
        fg(c);
        XSetLineAttributes(dpy_, gc_, width, LineSolid, CapButt, JoinMiter);
        XDrawLine(dpy_, d_, gc_, x1, y1, x2, y2);
    }
    void polyline(const std::vector<Pt>& p, Color c, int width = 1) {
        if (p.size() < 2)
            return;
        fg(c);
        XSetLineAttributes(dpy_, gc_, width, LineSolid, CapRound, JoinRound);
        std::vector<XPoint> xp(p.size());
        for (size_t i = 0; i < p.size(); ++i) {
            xp[i].x = (short)p[i].first;
            xp[i].y = (short)p[i].second;
        }
        XDrawLines(dpy_, d_, gc_, xp.data(), (int)xp.size(), CoordModeOrigin);
    }
    void text(int x, int y, const std::string& s, Color c) {  // (x,y) = top-left
        fg(c);
        XDrawString(dpy_, d_, gc_, x, y + (font_ ? font_->ascent : 11), s.c_str(), (int)s.size());
    }

   private:
    void fg(Color c) {
        unsigned key = ((unsigned)c.r << 16) | ((unsigned)c.g << 8) | c.b;
        auto it      = cache_.find(key);
        unsigned long px;
        if (it == cache_.end()) {
            XColor xc{};
            xc.red   = c.r * 257;
            xc.green = c.g * 257;
            xc.blue  = c.b * 257;
            xc.flags = DoRed | DoGreen | DoBlue;
            XAllocColor(dpy_, cmap_, &xc);
            px          = xc.pixel;
            cache_[key] = px;
        } else {
            px = it->second;
        }
        XSetForeground(dpy_, gc_, px);
    }
    Display* dpy_;
    Colormap cmap_;
    GC gc_;
    XFontStruct* font_;
    Drawable d_ = 0;
    MAP_<unsigned, unsigned long> cache_;
};

#endif

// (canvas, width, height, mouse_x, mouse_y)   mouse = -1,-1 when the pointer is outside
using DrawFn = std::function<void(Canvas&, int, int, int, int)>;

// ===========================================================================
//  Platform-independent plot rendering
// ===========================================================================
inline double nice_step(double range, int target) {
    if (!(range > 0) || !std::isfinite(range))
        return 1.0;
    double raw = range / target;
    double mag = std::pow(10.0, std::floor(std::log10(raw)));
    double f   = raw / mag;
    double nf  = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
    return nf * mag;
}

inline std::string fmt_g(double v) {
    char b[48];
    std::snprintf(b, sizeof b, "%g", v);
    return b;
}

// A named per-step value series (learning rate, grad norm, ...), see LivePlot::push_param().
// Series with the same non-zero `flag` share one panel; flag 0 means "its own panel".
struct Named {
    std::string name;
    int flag = 0;
    Series s;
};

struct Curve {
    const Series* s;
    std::string name, legend;
    Color col;
    bool markers;
};
struct Panel {
    std::vector<Curve> curves;
    int y0 = 0, y1 = 0;
    double ymin = 0, ymax = 1;
};

inline void draw_plot(Canvas& c, int W, int H, const Series& tr, const Series& ev, const std::vector<Named>& params, const std::string& title,
                      const std::string& status, int mx = -1, int my = -1) {
    const Color BG{255, 255, 255}, FG{30, 30, 30}, GRID{225, 225, 225}, GRAY{110, 110, 110}, BLUE{31, 119, 180}, ORANGE{255, 127, 14};
    static const Color PALETTE[] = {{44, 160, 44},   {214, 39, 40},   {148, 103, 189}, {140, 86, 75},
                                    {227, 119, 194}, {127, 127, 127}, {188, 189, 34},  {23, 190, 207}};
    c.fill_rect(0, 0, W, H, BG);

    const int th = c.text_h();
    const int L = 72, R = 24, B = th * 2 + 16;
    const int T   = 26 + 3 * th;
    const int px0 = L, px1 = W - R, top = T, bot = H - B;

    // ---- text header ----
    c.text(L, 8, title, FG);
    c.text(L, 8 + th + 4, status, GRAY);
    {
        char buf[200];
        std::string s;
        if (!tr.y.empty()) {
            std::snprintf(buf, sizeof buf, "step %.0f   train %.5f", tr.x.back(), tr.y.back());
            s += buf;
        }
        if (!ev.y.empty()) {
            std::snprintf(buf, sizeof buf, "   |   eval %.5f (step %.0f)", ev.y.back(), ev.x.back());
            s += buf;
        }
        c.text(L, 8 + 2 * (th + 4), s, FG);
    }

    if (px1 - px0 < 60 || bot - top < 60)
        return;

    // ---- panels: loss on top, then one per parameter (or per shared flag) ----
    std::vector<Panel> panels(1);
    panels[0].curves.push_back(Curve{&tr, "train", "train", BLUE, false});
    panels[0].curves.push_back(Curve{&ev, "eval", "eval", ORANGE, true});
    {
        std::map<std::string, size_t> where;  // panel key -> index in `panels`
        for (size_t i = 0; i < params.size(); ++i) {
            const Named& p = params[i];
            if (p.s.x.empty())
                continue;
            std::string key = p.flag == 0 ? "n:" + p.name : "f:" + std::to_string(p.flag);
            auto it         = where.find(key);
            if (it == where.end()) {
                it = where.emplace(key, panels.size()).first;
                panels.emplace_back();
            }
            char b[48];
            std::snprintf(b, sizeof b, "  %.4g", p.s.y.back());
            panels[it->second].curves.push_back(Curve{&p.s, p.name, p.name + b, PALETTE[i % 8], false});
        }
    }
    const int n = (int)panels.size();
    {
        const int GAP = th + 8;  // room for a legend strip above each param panel
        int avail     = (bot - top) - GAP * (n - 1);
        double wsum   = 2.0 + (n - 1);  // the loss panel is twice as tall
        int y         = top;
        for (int i = 0; i < n; ++i) {
            int h = (int)std::lround(avail * (i == 0 ? 2.0 : 1.0) / wsum);
            if (i == n - 1)
                h = bot - y;  // absorb rounding
            panels[i].y0 = y;
            panels[i].y1 = y + h;
            y += h + GAP;
        }
        for (auto& p : panels)
            if (p.y1 - p.y0 < 12)
                return;  // window too small for this many panels
    }
    auto frame = [&](const Panel& p) {
        c.line(px0, p.y0, px1, p.y0, FG);
        c.line(px0, p.y1, px1, p.y1, FG);
        c.line(px0, p.y0, px0, p.y1, FG);
        c.line(px1, p.y0, px1, p.y1, FG);
    };

    // ---- shared x range ----
    double xmin = 1e300, xmax = -1e300;
    for (auto& p : panels)
        for (auto& cu : p.curves)
            for (double v : cu.s->x) {
                xmin = std::min(xmin, v);
                xmax = std::max(xmax, v);
            }
    if (xmin > xmax) {  // no data at all yet
        c.text(px0 + 12, panels[0].y0 + 12, "waiting for data...", GRAY);
        frame(panels[0]);
        return;
    }
    if (xmax <= xmin)
        xmax = xmin + 1;

    // ---- per-panel y range ----
    for (auto& p : panels) {
        double lo = 1e300, hi = -1e300;
        for (auto& cu : p.curves)
            for (double v : cu.s->y) {
                lo = std::min(lo, v);
                hi = std::max(hi, v);
            }
        if (lo > hi) {
            lo = 0;
            hi = 1;
        }  // panel has no data yet
        if (hi <= lo) {
            double d = lo != 0 ? std::fabs(lo) * 0.1 : 0.5;
            lo -= d;
            hi += d;
        } else {
            double pad = (hi - lo) * 0.05;
            lo -= pad;
            hi += pad;
        }
        p.ymin = lo;
        p.ymax = hi;
    }
    auto X  = [&](double v) { return px0 + (int)std::lround((v - xmin) / (xmax - xmin) * (px1 - px0)); };
    auto Yp = [&](const Panel& p, double v) { return p.y1 - (int)std::lround((v - p.ymin) / (p.ymax - p.ymin) * (p.y1 - p.y0)); };

    // ---- grid, y tick labels, frame (every panel) ----
    const double xst    = nice_step(xmax - xmin, 7);
    const long long xk0 = (long long)std::ceil(xmin / xst), xk1 = (long long)std::floor(xmax / xst);
    for (const Panel& p : panels) {
        for (long long k = xk0; k <= xk1; ++k) {
            int px = X(k * xst);
            c.line(px, p.y0, px, p.y1, GRID);
        }
        double st    = nice_step(p.ymax - p.ymin, std::max(2, (p.y1 - p.y0) / 70));
        long long k0 = (long long)std::ceil(p.ymin / st), k1 = (long long)std::floor(p.ymax / st);
        for (long long k = k0; k <= k1; ++k) {
            double v = k * st;
            int py   = Yp(p, v);
            c.line(px0, py, px1, py, GRID);
            c.line(px0 - 4, py, px0, py, FG);
            std::string s = fmt_g(v);
            c.text(px0 - 7 - c.text_w(s), py - th / 2, s, FG);
        }
        frame(p);
    }
    // x tick marks + labels under the bottom panel only
    {
        const Panel& p = panels.back();
        for (long long k = xk0; k <= xk1; ++k) {
            double v = k * xst;
            int px   = X(v);
            c.line(px, p.y1, px, p.y1 + 4, FG);
            std::string s = fmt_g(v);
            c.text(px - c.text_w(s) / 2, p.y1 + 6, s, FG);
        }
        c.text((px0 + px1) / 2 - c.text_w("step") / 2, p.y1 + 6 + th + 4, "step", FG);
    }

    // ---- curves (decimated if there are more points than ~2 per pixel) ----
    for (const Panel& p : panels) {
        for (const Curve& cu : p.curves) {
            const Series& s = *cu.s;
            size_t cnt      = s.x.size();
            if (!cnt)
                continue;
            size_t maxpts = 2 * (size_t)(px1 - px0);
            std::vector<Pt> pts;
            if (cnt <= maxpts) {
                pts.reserve(cnt);
                for (size_t i = 0; i < cnt; ++i) pts.push_back({X(s.x[i]), Yp(p, s.y[i])});
            } else {
                pts.reserve(maxpts);
                for (size_t k = 0; k < maxpts; ++k) {
                    size_t i = k * (cnt - 1) / (maxpts - 1);
                    pts.push_back({X(s.x[i]), Yp(p, s.y[i])});
                }
            }
            if (pts.size() == 1)
                c.fill_rect(pts[0].first - 2, pts[0].second - 2, 5, 5, cu.col);
            else
                c.polyline(pts, cu.col, 2);
            if (cu.markers && cnt <= maxpts && cnt <= 200)
                for (auto& q : pts) c.fill_rect(q.first - 3, q.second - 3, 7, 7, cu.col);
        }
    }

    // ---- legends: parameter panels get a strip above the panel (never covers data) ----
    for (int pi = 1; pi < n; ++pi) {
        const Panel& p = panels[pi];
        int lx = px0, ly = p.y0 - th - 4;
        for (const Curve& cu : p.curves) {
            c.line(lx, ly + th / 2, lx + 22, ly + th / 2, cu.col, 2);
            c.text(lx + 28, ly, cu.legend, FG);
            lx += 28 + c.text_w(cu.legend) + 24;
        }
    }
    // ---- loss legend (top-right inside the loss panel) ----
    {
        const Panel& p = panels[0];
        int maxtw      = 0;
        for (const Curve& cu : p.curves) maxtw = std::max(maxtw, c.text_w(cu.legend));
        int eh = th + 6;
        int rw = 6 + 30 + maxtw + 8, rh = (int)p.curves.size() * eh + 4;
        int rx1 = px1 - 8, rx0 = rx1 - rw, ry0 = p.y0 + 5;
        c.fill_rect(rx0, ry0, rw, rh, BG);
        for (size_t i = 0; i < p.curves.size(); ++i) {
            int ly = ry0 + 5 + (int)i * eh, lx = rx0 + 6;
            c.line(lx, ly + th / 2, lx + 22, ly + th / 2, p.curves[i].col, 2);
            c.text(lx + 30, ly, p.curves[i].legend, FG);
        }
    }

    // ---- hover tooltip: nearest point on any curve of the panel under the mouse ----
    int hp = -1;
    if (mx >= px0 && mx <= px1)
        for (int i = 0; i < n; ++i)
            if (my >= panels[i].y0 && my <= panels[i].y1)
                hp = i;
    if (hp >= 0) {
        const Panel& P = panels[hp];
        struct Hit {
            const Curve* cu;
            size_t i;
            double d2;
        };
        Hit best{nullptr, 0, 1e18};
        // squared pixel distance from (px,py) to the segment (ax,ay)-(bx,by)
        auto seg_d2 = [](double px, double py, double ax, double ay, double bx, double by) {
            double vx = bx - ax, vy = by - ay, wx = px - ax, wy = py - ay;
            double len2 = vx * vx + vy * vy;
            double t    = len2 > 0 ? (wx * vx + wy * vy) / len2 : 0.0;
            t           = t < 0 ? 0 : (t > 1 ? 1 : t);
            double dx = wx - t * vx, dy = wy - t * vy;
            return dx * dx + dy * dy;
        };
        // Hit-test the drawn LINE (not just the data points), so hovering anywhere along
        // it works even when points are far apart; snap to the nearer end of that segment.
        for (const Curve& cu : P.curves) {
            const Series& s = *cu.s;
            double ax = 0, ay = 0;
            for (size_t i = 0; i < s.x.size(); ++i) {
                double bx = X(s.x[i]), by = Yp(P, s.y[i]);
                double db2 = (bx - mx) * (bx - mx) + (by - my) * (by - my);
                if (i == 0) {
                    if (db2 < best.d2)
                        best = Hit{&cu, 0, db2};
                } else {
                    double sd2 = seg_d2(mx, my, ax, ay, bx, by);
                    if (sd2 < best.d2) {
                        double da2 = (ax - mx) * (ax - mx) + (ay - my) * (ay - my);
                        best       = Hit{&cu, da2 <= db2 ? i - 1 : i, sd2};
                    }
                }
                ax = bx;
                ay = by;
            }
        }
        const double R_HIT = 12.0;  // pick radius in pixels
        if (best.cu && best.d2 <= R_HIT * R_HIT) {
            const Series& s = *best.cu->s;
            double sx = s.x[best.i], sy = s.y[best.i];
            int hx = X(sx), hy = Yp(P, sy);
            c.line(hx, panels.front().y0, hx, panels.back().y1, GRAY);  // guide through all panels
            c.fill_rect(hx - 5, hy - 5, 11, 11, FG);                    // highlight marker
            c.fill_rect(hx - 4, hy - 4, 9, 9, best.cu->col);

            std::string l1 = best.cu->name;
            std::string l2 = "step " + fmt_g(sx);
            char b[48];
            std::snprintf(b, sizeof b, hp == 0 ? "loss %.6g" : "value %.6g", sy);
            std::string l3 = b;
            int bw         = std::max(c.text_w(l1), std::max(c.text_w(l2), c.text_w(l3))) + 16;
            int bh         = 3 * th + 16;
            int bx = hx + 14, by = hy - bh - 10;  // prefer upper-right of the point
            if (bx + bw > W - 2)
                bx = hx - 14 - bw;  // flip left if it would overflow
            if (bx < 2)
                bx = 2;
            if (by < 2)
                by = hy + 14;  // flip below
            if (by + bh > H - 2)
                by = H - 2 - bh;
            c.fill_rect(bx - 1, by - 1, bw + 2, bh + 2, FG);    // border
            c.fill_rect(bx, by, bw, bh, Color{255, 255, 225});  // background
            c.text(bx + 8, by + 8, l1, best.cu->col);
            c.text(bx + 8, by + 8 + th, l2, FG);
            c.text(bx + 8, by + 8 + 2 * th, l3, FG);
        }
    }
}

// ===========================================================================
//  Platform layer: window + event loop (runs on the plot thread)
// ===========================================================================
#ifdef _WIN32

struct Ctx {
    const DrawFn* draw;
    Shared* sh;
    int mx = -1, my = -1;   // mouse position, -1 = outside window
    bool tracking = false;  // TrackMouseEvent armed (to get WM_MOUSELEAVE)
};

inline LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    Ctx* ctx = (Ctx*)GetWindowLongPtrA(h, GWLP_USERDATA);
    switch (m) {
        case WM_NCCREATE:
            SetWindowLongPtrA(h, GWLP_USERDATA, (LONG_PTR)((CREATESTRUCTA*)l)->lpCreateParams);
            break;
        case WM_ERASEBKGND:
            return 1;
        case WM_SIZE:
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_APP + 1:  // sent by LivePlot::wake() from any thread
            if (ctx && ctx->sh->stop)
                DestroyWindow(h);
            else
                InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_KEYDOWN:
            if (w == VK_ESCAPE)
                DestroyWindow(h);
            return 0;
        case WM_MOUSEMOVE:
            if (ctx) {
                ctx->mx = (short)LOWORD(l);
                ctx->my = (short)HIWORD(l);
                if (!ctx->tracking) {
                    TRACKMOUSEEVENT t{sizeof t, TME_LEAVE, h, 0};
                    TrackMouseEvent(&t);
                    ctx->tracking = true;
                }
                InvalidateRect(h, nullptr, FALSE);
            }
            return 0;
        case WM_MOUSELEAVE:
            if (ctx) {
                ctx->mx = ctx->my = -1;
                ctx->tracking     = false;
            }
            InvalidateRect(h, nullptr, FALSE);
            return 0;
        case WM_PAINT: {
            PAINTSTRUCT ps;
            HDC hdc = BeginPaint(h, &ps);
            RECT rc;
            GetClientRect(h, &rc);
            int W = rc.right, H = rc.bottom;
            if (ctx && W > 0 && H > 0) {
                HDC mem     = CreateCompatibleDC(hdc);
                HBITMAP bmp = CreateCompatibleBitmap(hdc, W, H);
                HGDIOBJ old = SelectObject(mem, bmp);
                {
                    Canvas cv(mem);
                    (*ctx->draw)(cv, W, H, ctx->mx, ctx->my);
                }
                BitBlt(hdc, 0, 0, W, H, mem, 0, 0, SRCCOPY);  // double-buffered, no flicker
                SelectObject(mem, old);
                DeleteObject(bmp);
                DeleteDC(mem);
            }
            EndPaint(h, &ps);
            return 0;
        }
        case WM_DESTROY:
            if (ctx && ctx->sh->want_png) {  // capture one last clean frame for stop() to save
                RECT rc;
                GetClientRect(h, &rc);
                int W = rc.right, H = rc.bottom;
                if (W > 0 && H > 0) {
                    BITMAPINFO bi{};
                    bi.bmiHeader.biSize        = sizeof(BITMAPINFOHEADER);
                    bi.bmiHeader.biWidth       = W;
                    bi.bmiHeader.biHeight      = -H;  // top-down
                    bi.bmiHeader.biPlanes      = 1;
                    bi.bmiHeader.biBitCount    = 32;
                    bi.bmiHeader.biCompression = BI_RGB;
                    void* bits                 = nullptr;
                    HDC mem                    = CreateCompatibleDC(nullptr);
                    HBITMAP bmp                = CreateDIBSection(mem, &bi, DIB_RGB_COLORS, &bits, nullptr, 0);
                    if (bmp && bits) {
                        HGDIOBJ old = SelectObject(mem, bmp);
                        {
                            Canvas cv(mem);
                            (*ctx->draw)(cv, W, H, -1, -1);
                        }
                        GdiFlush();
                        const unsigned char* p          = (const unsigned char*)bits;  // BGRA
                        std::vector<unsigned char>& out = ctx->sh->cap_rgb;
                        out.resize((size_t)W * H * 3);
                        for (size_t i = 0, n = (size_t)W * H; i < n; ++i) {
                            out[3 * i]     = p[4 * i + 2];
                            out[3 * i + 1] = p[4 * i + 1];
                            out[3 * i + 2] = p[4 * i];
                        }
                        ctx->sh->cap_w = W;
                        ctx->sh->cap_h = H;
                        SelectObject(mem, old);
                    }
                    if (bmp)
                        DeleteObject(bmp);
                    DeleteDC(mem);
                }
            }
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcA(h, m, w, l);
}

inline void run_window(const std::string& title, int w, int h, Shared& sh, const DrawFn& draw) {
    HINSTANCE hi = GetModuleHandleA(nullptr);
    WNDCLASSA wc{};
    wc.lpfnWndProc   = WndProc;
    wc.hInstance     = hi;
    wc.hCursor       = LoadCursorA(nullptr, (LPCSTR)IDC_ARROW);
    wc.lpszClassName = "LivePlotWindow";
    RegisterClassA(&wc);  // harmless if already registered
    RECT r{0, 0, w, h};
    AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
    Ctx ctx{&draw, &sh};
    HWND hwnd = CreateWindowA("LivePlotWindow", title.c_str(), WS_OVERLAPPEDWINDOW | WS_VISIBLE, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left,
                              r.bottom - r.top, nullptr, nullptr, hi, &ctx);
    if (!hwnd) {
        std::fprintf(stderr, "[LivePlot] CreateWindow failed\n");
        return;
    }
    sh.native = hwnd;
    if (sh.stop)
        DestroyWindow(hwnd);  // stop() raced with start()
    MSG msg;
    while (GetMessageA(&msg, nullptr, 0, 0) > 0) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    sh.native = nullptr;
}

#else  // ---------------------------- X11 ------------------------------------

inline void run_window(const std::string& title, int w, int h, Shared& sh, const DrawFn& draw) {
    Display* dpy = XOpenDisplay(nullptr);
    if (!dpy) {
        std::fprintf(stderr, "[LivePlot] cannot open X display (is $DISPLAY set?)\n");
        return;
    }
    int scr      = DefaultScreen(dpy);
    ::Window win = XCreateSimpleWindow(dpy, RootWindow(dpy, scr), 0, 0, w, h, 0, BlackPixel(dpy, scr), WhitePixel(dpy, scr));
    XStoreName(dpy, win, title.c_str());
    XSelectInput(dpy, win, ExposureMask | StructureNotifyMask | KeyPressMask | PointerMotionMask | LeaveWindowMask);
    Atom wmdel = XInternAtom(dpy, "WM_DELETE_WINDOW", False);
    XSetWMProtocols(dpy, win, &wmdel, 1);
    XMapWindow(dpy, win);

    GC gc = XCreateGC(dpy, win, 0, nullptr);
    XSetGraphicsExposures(dpy, gc, False);
    XFontStruct* font = XLoadQueryFont(dpy, "-*-fixed-medium-r-normal--13-*-*-*-*-*-iso8859-1");
    if (!font)
        font = XLoadQueryFont(dpy, "fixed");
    if (font)
        XSetFont(dpy, gc, font->fid);

    Canvas cv(dpy, DefaultColormap(dpy, scr), gc, font);
    Pixmap pm = 0;
    int pw = 0, ph = 0;
    bool closed = false, need = true;
    int mx = -1, my = -1;  // mouse position, -1 = outside window
    const int fd = ConnectionNumber(dpy);

    // Only this thread touches the Display, so no XInitThreads needed. Other threads just
    // set sh.dirty; we poll it every <=30 ms while idling on the X connection.
    while (!closed && !sh.stop) {
        while (XPending(dpy)) {
            XEvent e;
            XNextEvent(dpy, &e);
            if (e.type == Expose)
                need = true;
            else if (e.type == MotionNotify) {
                mx   = e.xmotion.x;
                my   = e.xmotion.y;
                need = true;
            } else if (e.type == LeaveNotify) {
                mx = my = -1;
                need    = true;
            } else if (e.type == ConfigureNotify) {
                w    = e.xconfigure.width;
                h    = e.xconfigure.height;
                need = true;
            } else if (e.type == ClientMessage) {
                if ((Atom)e.xclient.data.l[0] == wmdel)
                    closed = true;
            } else if (e.type == KeyPress) {
                if (XLookupKeysym(&e.xkey, 0) == XK_Escape)
                    closed = true;
            }
        }
        if (sh.dirty.exchange(false))
            need = true;
        if (need && w > 0 && h > 0) {
            need = false;
            if (!pm || w != pw || h != ph) {
                if (pm)
                    XFreePixmap(dpy, pm);
                pm = XCreatePixmap(dpy, win, w, h, DefaultDepth(dpy, scr));
                pw = w;
                ph = h;
            }
            cv.target(pm);
            draw(cv, w, h, mx, my);  // off-screen -> no flicker
            XCopyArea(dpy, pm, win, gc, 0, 0, w, h, 0, 0);
            XFlush(dpy);
        }
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(fd, &fds);
        timeval tv{0, 30000};
        select(fd + 1, &fds, nullptr, nullptr, &tv);
    }
    // ---- capture one last clean frame (no tooltip) for stop() to save as PNG ----
    if (sh.want_png && w > 0 && h > 0) {
        if (!pm || w != pw || h != ph) {
            if (pm)
                XFreePixmap(dpy, pm);
            pm = XCreatePixmap(dpy, win, w, h, DefaultDepth(dpy, scr));
            pw = w;
            ph = h;
        }
        cv.target(pm);
        draw(cv, w, h, -1, -1);
        XImage* im = XGetImage(dpy, pm, 0, 0, w, h, AllPlanes, ZPixmap);
        if (im) {
            auto chan = [](unsigned long p, unsigned long m) -> unsigned char {
                int s            = 0;
                unsigned long mm = m;
                while (mm && !(mm & 1)) {
                    mm >>= 1;
                    ++s;
                }
                return mm ? (unsigned char)(((p & m) >> s) * 255 / mm) : 0;
            };
            // A Pixmap has no visual, so im->red_mask etc. are 0: take the masks from the screen's visual.
            Visual* vis       = DefaultVisual(dpy, scr);
            const bool direct = (vis->c_class == TrueColor || vis->c_class == DirectColor) && vis->red_mask;
            std::map<unsigned long, unsigned> lut;  // palette visuals (e.g. 8-bit): pixel -> 0xRRGGBB
            if (!direct) {
                for (int y = 0; y < h; ++y)
                    for (int x = 0; x < w; ++x) lut[XGetPixel(im, x, y)] = 0;
                std::vector<XColor> cols;
                for (auto& kv : lut) {
                    XColor c{};
                    c.pixel = kv.first;
                    cols.push_back(c);
                }
                if (!cols.empty())
                    XQueryColors(dpy, DefaultColormap(dpy, scr), cols.data(), (int)cols.size());
                for (auto& c : cols) lut[c.pixel] = ((unsigned)(c.red >> 8) << 16) | ((unsigned)(c.green >> 8) << 8) | (unsigned)(c.blue >> 8);
            }
            sh.cap_rgb.resize((size_t)w * h * 3);
            size_t k = 0;
            bool any = false;
            for (int y = 0; y < h; ++y)
                for (int x = 0; x < w; ++x) {
                    unsigned long p = XGetPixel(im, x, y);
                    unsigned char r, g, b;
                    if (direct) {
                        r = chan(p, vis->red_mask);
                        g = chan(p, vis->green_mask);
                        b = chan(p, vis->blue_mask);
                    } else {
                        unsigned v = lut[p];
                        r          = (unsigned char)(v >> 16);
                        g          = (unsigned char)(v >> 8);
                        b          = (unsigned char)v;
                    }
                    any             = any || (r | g | b);
                    sh.cap_rgb[k++] = r;
                    sh.cap_rgb[k++] = g;
                    sh.cap_rgb[k++] = b;
                }
            if (!any)
                std::fprintf(stderr,
                             "[LivePlot] warning: captured image is all black "
                             "(depth %d, visual class %d, masks %lx/%lx/%lx)\n",
                             DefaultDepth(dpy, scr), vis->c_class, vis->red_mask, vis->green_mask, vis->blue_mask);
            sh.cap_w = w;
            sh.cap_h = h;
            XDestroyImage(im);
        }
    }
    if (pm)
        XFreePixmap(dpy, pm);
    XFreeGC(dpy, gc);
    if (font)
        XFreeFont(dpy, font);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
}

#endif

}  // namespace liveplot

// ===========================================================================
//  Public API
// ===========================================================================
class LivePlot {
   public:
    explicit LivePlot(std::string title = "Training curves", int width = 900, int height = 560) : title_(std::move(title)), w_(width), h_(height) {}
    ~LivePlot() { stop(); }
    LivePlot(const LivePlot&)            = delete;
    LivePlot& operator=(const LivePlot&) = delete;

    void start() {
        if (thread_.joinable())
            return;
        sh_.stop = false;
        running_ = true;
        thread_  = std::thread([this] { run(); });
    }
    // Call before start(). When stop() runs (also from the destructor, or after you closed the
    // window by hand), the final curves are saved to this PNG file.
    void save_png_on_stop(const std::string& path) {
        png_path_    = path;
        sh_.want_png = true;
    }

    void stop() {
        sh_.stop = true;
        wake();
        if (thread_.joinable())
            thread_.join();
        if (!png_path_.empty() && !sh_.cap_rgb.empty()) {  // window thread has exited: safe to read
            if (liveplot::write_png(png_path_, sh_.cap_w, sh_.cap_h, sh_.cap_rgb))
                std::fprintf(stderr, "[LivePlot] saved %s\n", png_path_.c_str());
            else
                std::fprintf(stderr, "[LivePlot] cannot write %s\n", png_path_.c_str());
            sh_.cap_rgb.clear();
        }
    }

    void push_train(double step, double loss) { push(train_, step, loss); }
    void push_eval(double step, double loss) { push(eval_, step, loss); }

    // Plot any named per-step value (learning rate, grad norm, ...) in a panel under the loss plot.
    //   flag == 0 : this parameter gets its own panel (own y-axis)
    //   flag != 0 : all parameters pushed with the same flag share one panel / y-axis
    // Panels appear in order of the first push. Thread-safe; NaN/inf values are ignored.
    void push_param(int step, const std::string& param_name, float value, int flag = 0) {
        if (!std::isfinite(value))
            return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto it = param_idx_.find(param_name);
            size_t i;
            if (it == param_idx_.end()) {
                i = params_.size();
                param_idx_.emplace(param_name, i);
                liveplot::Named nm;
                nm.name = param_name;
                params_.push_back(std::move(nm));
            } else {
                i = it->second;
            }
            params_[i].flag = flag;
            params_[i].s.x.push_back(step);
            params_[i].s.y.push_back(value);
        }
        wake();
    }
    void set_status(const std::string& s) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            status_ = s;
        }
        wake();
    }
    bool is_open() const { return running_.load(); }

   private:
    //  Never waits for the screen/drawing.It takes a mutex & sets an atomic “dirty” flag
    void push(liveplot::Series& s, double step, double loss) {
        if (!std::isfinite(step) || !std::isfinite(loss))
            return;
        {
            std::lock_guard<std::mutex> lk(mu_);
            s.x.push_back(step);
            s.y.push_back(loss);
        }
        wake();
    }
    void wake() {
        sh_.dirty = true;
#ifdef _WIN32
        if (void* h = sh_.native.load())
            PostMessageA((HWND)h, WM_APP + 1, 0, 0);
#endif
    }
    void run() {
        liveplot::DrawFn fn = [this](liveplot::Canvas& c, int W, int H, int mx, int my) {
            liveplot::Series tr, ev;
            std::vector<liveplot::Named> pr;
            std::string st;
            {
                std::lock_guard<std::mutex> lk(mu_);
                tr = train_;
                ev = eval_;
                pr = params_;
                st = status_;
            }
            liveplot::draw_plot(c, W, H, tr, ev, pr, title_, st, mx, my);  // snapshot: trainer never blocks on drawing
        };
        liveplot::run_window(title_, w_, h_, sh_, fn);
        running_ = false;
    }

    std::string title_;
    int w_, h_;
    std::thread thread_;
    std::atomic<bool> running_{false};
    liveplot::Shared sh_;
    std::mutex mu_;
    liveplot::Series train_, eval_;
    std::vector<liveplot::Named> params_;                // extra per-step values, in first-push order
    MAP_<std::string, size_t> param_idx_;  // name -> index in params_
    std::string status_;
    std::string png_path_;
};
typedef std::shared_ptr<LivePlot> hLivePlot;