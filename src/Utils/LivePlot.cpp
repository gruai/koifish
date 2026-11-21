/**
 *  SPDX-FileCopyrightText: Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  \brief Plot functions
 *  \author Yingshi Chen
 */
#include "LivePlot.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>

int demp() {
    LivePlot plot("My LLM training");
    plot.start();

    const int total_steps = 5000;
    const int eval_every  = 50;

    for (int step = 1; step <= total_steps && plot.is_open(); ++step) {
        // ---- your real training step goes here ----
        double noise      = (std::rand() / (double)RAND_MAX - 0.5) * 0.1;
        double train_loss = 4.0 * std::exp(-step / 800.0) + 0.5 + noise;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
        // -------------------------------------------

        plot.push_train(step, train_loss);

        if (step % 20 == 0) {
            char buf[128];
            std::snprintf(buf, sizeof buf, "lr=3e-4   batch=64   %d/%d steps", step, total_steps);
            plot.set_status(buf);
        }
        if (step % eval_every == 0) {
            double eval_loss = 4.2 * std::exp(-step / 900.0) + 0.6;
            plot.push_eval(step, eval_loss);
        }
    }

    // Keep the window open after training ends until the user closes it:
    // while (plot.is_open()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return 0;
}

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define getpid _getpid
#else
#include <unistd.h>
#endif

using namespace std::chrono_literals;
namespace fs = std::filesystem;

// ------------------------------------------------------------------ tiny JSON (flat objects only) ----
struct Val {
    std::string text;
    bool quoted = false;
    Val(int x) : text(std::to_string(x)) {}
    Val(long x) : text(std::to_string(x)) {}
    Val(double x) {
        char b[64];
        std::snprintf(b, sizeof b, "%.6f", x);
        text = b;
    }
    Val(bool x) : text(x ? "true" : "false") {}
    Val(const char* s) : text(s), quoted(true) {}
    Val(const std::string& s) : text(s), quoted(true) {}
};

struct Json {
    std::vector<std::pair<std::string, Val>> kv;
    Json() = default;
    Json(std::initializer_list<std::pair<std::string, Val>> il) : kv(il) {}
    static std::string esc(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '"' || c == '\\') {
                o += '\\';
                o += c;
            } else if (c == '\n')
                o += "\\n";
            else
                o += c;
        }
        return o;
    }
    std::string dump() const {
        std::string o = "{";
        for (size_t i = 0; i < kv.size(); ++i) {
            if (i)
                o += ", ";
            o += "\"" + esc(kv[i].first) + "\": ";
            o += kv[i].second.quoted ? "\"" + esc(kv[i].second.text) + "\"" : kv[i].second.text;
        }
        return o + "}";
    }
};

// ------------------------------------------------------------------ collaborator interfaces ----
struct IClosable {
    virtual ~IClosable() = default;
    virtual void close() = 0;
};  // httpd.shutdown / vision.shutdown / hub.close

struct IEngine : IClosable {
    virtual bool has_proc() const = 0;
    virtual void kill_proc()      = 0;
    virtual void drop_cache()     = 0;  // cache_drop.drop_after_exit(engine); may throw
};

struct IService {
    virtual ~IService()                          = default;
    virtual std::string instance_id() const      = 0;
    virtual std::string model() const            = 0;
    virtual int begin_drain()                    = 0;  // returns pending request count
    virtual int wait_for_drain(double timeout_s) = 0;  // returns requests still pending
    virtual int cancel_owned_requests()          = 0;
};

struct IReadyFile {
    virtual ~IReadyFile()                                                                 = default;
    virtual bool publish(const std::string& path, const Json& record)                     = 0;  // false or throw on I/O error
    virtual void remove_if_owned(const std::string& path, const std::string& instance_id) = 0;
};

using Notify = std::function<void(const std::string&, const Json&)>;  // notify_lifecycle

struct Args {
    bool open        = false;
    std::string host = "127.0.0.1";
    std::optional<double> shutdown_timeout;
};

struct Ctx {
    Args args;
    std::optional<bool> cfg_open_browser;  // cfg.get("open_browser")  (nullopt = key absent)
    std::string here  = "127.0.0.1";
    int bound_port    = 0;
    IService* svc     = nullptr;  // required
    IEngine* engine   = nullptr;  // required
    IClosable* httpd  = nullptr;  // optional
    IClosable* vision = nullptr;  // optional
    IClosable* hub    = nullptr;  // optional
    std::optional<std::string> ready_path;
    IReadyFile* ready_file      = nullptr;  // required when ready_path is set
    bool has_event_sink         = false;
    bool lazy                   = false;
    int interrupt_exit_code     = 130;
    std::atomic<bool>* finished = nullptr;  // optional
    Notify notify;                          // optional
};

// ------------------------------------------------------------------ signals ----
static std::atomic<int> g_sigint_count{0};
static std::atomic<int> g_sigterm{0};

extern "C" void on_signal(int sig) {  // async-signal-safe: only lock-free atomics
    if (sig == SIGINT)
        g_sigint_count.fetch_add(1);
    else
        g_sigterm.store(1);
}

// ------------------------------------------------------------------ browser ----
static std::function<void(const std::string&)> g_browser_opener = [](const std::string& url) {
#if defined(_WIN32)
    std::string cmd = "start \"\" \"" + url + "\"";
#elif defined(__APPLE__)
    std::string cmd = "open \"" + url + "\" >/dev/null 2>&1";
#else
    std::string cmd = "xdg-open \"" + url + "\" >/dev/null 2>&1";
#endif
    int rc = std::system(cmd.c_str());
    (void)rc;
};

// ------------------------------------------------------------------ main tail ----
int run_tail(Ctx& c) {
    if (!c.svc || !c.engine) {
        std::fprintf(stderr, "[strata] run_tail: svc and engine are required\n");
        return 2;
    }
    if (c.ready_path && !c.ready_file) {
        std::fprintf(stderr, "[strata] run_tail: ready_file missing\n");
        return 2;
    }

    auto notify = [&](const std::string& ev, const Json& j) {
        if (c.notify)
            c.notify(ev, j);
    };

    g_sigint_count = 0;  // fresh state for every run
    g_sigterm      = 0;
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // 1) open the browser: config's "open_browser": false wins (#609); own thread so signals stay serviceable (#1872)
    if (c.args.open && c.cfg_open_browser.value_or(true)) {
        std::string url = "http://" + c.here + ":" + std::to_string(c.bound_port) + "/";
        std::thread([url] {
            try {
                g_browser_opener(url);
            } catch (...) {
            }
        }).detach();
    }

    Json record;
    if (c.ready_path || c.has_event_sink) {
        std::string ready_host = (c.here.find(':') != std::string::npos && c.here.rfind('[', 0) != 0) ? "[" + c.here + "]" : c.here;
        record                 = Json{
            {"schema_version", 1},
            {"instance_id", c.svc->instance_id()},
            {"pid", static_cast<long>(getpid())},
            {"api_url", "http://" + ready_host + ":" + std::to_string(c.bound_port)},
            {"bound_host", c.args.host},
            {"bound_port", c.bound_port},
            {"model_id", c.svc->model()},
            {"model_state", c.lazy ? "unloaded" : "ready"},
            {"api_state", "ready"},
            {"created_at", std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count()},
        };
    }

    bool interrupted = false, drained = true;
    int cancelled           = 0;
    std::string termination = "sigint";
    int baseline            = 0;

    // the Python `finally:` block (a lambda so the early `return 6` can run it as well)
    auto finally_block = [&] {
        if (c.args.shutdown_timeout) {
            double t    = *c.args.shutdown_timeout;
            int pending = c.svc->begin_drain();
            notify("draining", Json{{"instance_id", c.svc->instance_id()}, {"pending_requests", pending}, {"deadline_s", t}});
            if (pending) {
                std::printf("[strata] draining %d inference call(s) for up to %g s ...\n", pending, t);
                std::fflush(stdout);
            }
            pending = c.svc->wait_for_drain(t);
            if (pending) {
                cancelled = c.svc->cancel_owned_requests();
                std::printf("[strata] drain deadline expired; cancelling %d owned request(s) ...\n", cancelled);
                std::fflush(stdout);
                c.svc->wait_for_drain(5);
                drained = false;
            }
        }

        std::vector<IClosable*> closers;
        if (c.httpd)
            closers.push_back(c.httpd);
        closers.push_back(c.engine);
        if (c.vision)
            closers.push_back(c.vision);
        if (c.hub)
            closers.push_back(c.hub);

        // closers run on a worker so this thread can still see a SECOND Ctrl+C. An exception must never escape
        // a std::thread (that is std::terminate), so each closer is guarded.
        std::atomic<bool> done{false};
        std::thread worker([&] {
            for (IClosable* cl : closers) {
                try {
                    cl->close();
                } catch (...) { /* keep closing the rest */
                }
            }
            done = true;
        });

        bool killed = false;
        while (!done) {
            if (!killed && g_sigint_count.load() > baseline) {  // second Ctrl+C: don't wait for the engine
                killed = true;
                if (c.engine->has_proc()) {
                    try {
                        c.engine->kill_proc();
                        c.engine->drop_cache();  // opt-in, never blocks the exit
                    } catch (const std::exception& e) {
                        std::printf("[strata] could not release the model files' cache: %s\n", e.what());
                        std::fflush(stdout);
                    } catch (...) {
                    }
                }
            }
            std::this_thread::sleep_for(20ms);
        }
        worker.join();

        if (c.ready_path) {
            try {
                c.ready_file->remove_if_owned(*c.ready_path, c.svc->instance_id());
            } catch (...) {
            }
        }
        if (c.finished)
            *c.finished = true;
        if (interrupted) {
            std::puts("[strata] stopped");
            std::fflush(stdout);
        }
    };

    // 2) serve: publish ready, then idle until a signal arrives
    if (c.ready_path) {
        bool ok = false;
        try {
            ok = c.ready_file->publish(*c.ready_path, record);
        } catch (...) {
            ok = false;
        }
        if (!ok) {
            std::fprintf(stderr, "[strata] could not write ready file (OSError)\n");
            finally_block();  // closes everything once
            return 6;
        }
    }
    if (c.has_event_sink)
        notify("ready", record);

    while (g_sigint_count.load() == 0 && g_sigterm.load() == 0) std::this_thread::sleep_for(100ms);

    // 3) interrupted
    interrupted = true;
    baseline    = g_sigint_count.load();
    termination = (baseline > 0) ? "sigint" : "sigterm";
    notify("stopping", Json{{"reason", termination}, {"instance_id", c.svc->instance_id()}});
    std::puts("\n[strata] stopping (Ctrl+C again to end the engine at once) ...");
    std::fflush(stdout);

    finally_block();

    int exit_code = !drained ? 8 : (interrupted && termination == "sigint") ? c.interrupt_exit_code : 0;
    notify("stopped", Json{{"instance_id", c.svc->instance_id()},
                           {"reason", interrupted ? termination : std::string("normal")},
                           {"drained", drained},
                           {"cancelled_requests", cancelled},
                           {"exit_code", exit_code}});
    return exit_code;
}

// =================================================================== test harness ===================================================================
namespace {

struct MockService : IService {
    double drain_work_s  = 0.3;  // how long in-flight requests need to finish
    int pending_at_start = 2;
    std::atomic<int> pending{0};
    std::string instance_id() const override { return "test-instance"; }
    std::string model() const override { return "test-model"; }
    int begin_drain() override {
        pending = pending_at_start;
        return pending;
    }
    int wait_for_drain(double timeout_s) override {
        double wait = std::min(timeout_s, drain_work_s);
        std::this_thread::sleep_for(std::chrono::duration<double>(wait));
        if (drain_work_s <= timeout_s)
            pending = 0;
        drain_work_s = std::max(0.0, drain_work_s - wait);
        return pending;
    }
    int cancel_owned_requests() override {
        int n        = pending;
        pending      = 0;
        drain_work_s = 0;
        return n;
    }
};

struct MockClosable : IClosable {
    std::atomic<bool> closed{false};
    bool throws = false;
    void close() override {
        closed = true;
        if (throws)
            throw std::runtime_error("close failed");
    }
};

struct MockEngine : IEngine {
    std::atomic<bool> closed{false}, killed{false}, cache_dropped{false};
    double close_blocks_s = 0;  // simulates an engine that takes long to free memory
    void close() override {
        auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(close_blocks_s);
        while (!killed && std::chrono::steady_clock::now() < end) std::this_thread::sleep_for(10ms);
        closed = true;
    }
    bool has_proc() const override { return true; }
    void kill_proc() override { killed = true; }
    void drop_cache() override { cache_dropped = true; }
};

struct FileReady : IReadyFile {
    bool fail     = false;
    int publishes = 0, removes = 0;
    bool publish(const std::string& path, const Json& record) override {
        ++publishes;
        if (fail)
            return false;
        std::ofstream f(path);
        if (!f)
            return false;
        f << record.dump() << "\n";
        return true;
    }
    void remove_if_owned(const std::string& path, const std::string&) override {
        ++removes;
        std::error_code ec;
        fs::remove(path, ec);
    }
};

int g_failures = 0;
#define CHECK(cond)                                                    \
    do {                                                               \
        if (!(cond)) {                                                 \
            ++g_failures;                                              \
            std::printf("    FAIL: %s  (line %d)\n", #cond, __LINE__); \
        }                                                              \
    } while (0)

struct Scenario {
    MockService svc;
    MockEngine engine;
    MockClosable httpd, vision, hub;
    FileReady rf;
    std::vector<std::string> events;
    std::atomic<bool> finished{false};
    std::string ready_path = (fs::temp_directory_path() / "strata_ready_test.json").string();
    Ctx ctx;

    Scenario() {
        ctx.svc            = &svc;
        ctx.engine         = &engine;
        ctx.httpd          = &httpd;
        ctx.vision         = &vision;
        ctx.hub            = &hub;
        ctx.ready_file     = &rf;
        ctx.ready_path     = ready_path;
        ctx.has_event_sink = true;
        ctx.finished       = &finished;
        ctx.bound_port     = 8080;
        ctx.notify         = [this](const std::string& ev, const Json& j) {
            events.push_back(ev);
            std::printf("    event %-9s %s\n", ev.c_str(), j.dump().c_str());
        };
        std::error_code ec;
        fs::remove(ready_path, ec);
    }
    bool all_closed() const { return httpd.closed && engine.closed && vision.closed && hub.closed; }
};

// raise signals from a helper thread at the given offsets (seconds)
std::thread raiser(std::vector<std::pair<double, int>> plan) {
    return std::thread([plan] {
        double last = 0;
        for (auto& [at, sig] : plan) {
            std::this_thread::sleep_for(std::chrono::duration<double>(at - last));
            last = at;
            std::raise(sig);
        }
    });
}

double secs_since(std::chrono::steady_clock::time_point t0) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }

void test_clean_ctrl_c() {
    std::puts("[1] Ctrl+C, drain finishes in time");
    Scenario s;
    s.ctx.args.shutdown_timeout = 5.0;
    auto t                      = raiser({{0.3, SIGINT}});
    int rc                      = run_tail(s.ctx);
    t.join();
    CHECK(rc == 130);
    CHECK((s.events == std::vector<std::string>{"ready", "stopping", "draining", "stopped"}));
    CHECK(s.rf.publishes == 1 && s.rf.removes == 1);
    CHECK(!fs::exists(s.ready_path));
    CHECK(s.all_closed());
    CHECK(s.finished);
    CHECK(!s.engine.killed);
}

void test_drain_timeout() {
    std::puts("[2] Ctrl+C, drain deadline expires -> requests cancelled, exit code 8");
    Scenario s;
    s.ctx.args.shutdown_timeout = 0.4;
    s.svc.drain_work_s          = 3.0;
    auto t                      = raiser({{0.3, SIGINT}});
    int rc                      = run_tail(s.ctx);
    t.join();
    CHECK(rc == 8);
    CHECK(s.svc.pending == 0);
    CHECK(s.all_closed());
}

void test_second_ctrl_c() {
    std::puts("[3] second Ctrl+C while the engine is closing -> engine killed at once");
    Scenario s;
    s.engine.close_blocks_s = 10.0;
    auto t0                 = std::chrono::steady_clock::now();
    auto t                  = raiser({{0.3, SIGINT}, {2.0, SIGINT}});
    int rc                  = run_tail(s.ctx);
    t.join();
    double el = secs_since(t0);
    std::printf("    elapsed %.2f s\n", el);
    CHECK(rc == 130);
    CHECK(s.engine.killed);
    CHECK(s.engine.cache_dropped);
    CHECK(el < 6.0);  // did NOT wait the full 10 s
    CHECK(s.all_closed());
}

void test_ready_file_failure() {
    std::puts("[4] ready file cannot be written -> exit 6, everything closed once");
    Scenario s;
    s.rf.fail = true;
    int rc    = run_tail(s.ctx);
    CHECK(rc == 6);
    CHECK(s.all_closed());
    CHECK(s.finished);
    CHECK(s.rf.removes == 1);
    CHECK(s.events.empty());  // no "ready", no "stopped"
}

void test_sigterm() {
    std::puts("[5] SIGTERM -> clean stop, exit 0");
    Scenario s;
    auto t = raiser({{0.3, SIGTERM}});
    int rc = run_tail(s.ctx);
    t.join();
    CHECK(rc == 0);
    CHECK(s.all_closed());
    CHECK(!s.events.empty() && s.events.back() == "stopped");
}

void test_throwing_closer() {
    std::puts("[6] a closer that throws must not crash (no std::terminate) and others still run");
    Scenario s;
    s.httpd.throws = true;
    auto t         = raiser({{0.3, SIGINT}});
    int rc         = run_tail(s.ctx);
    t.join();
    CHECK(rc == 130);
    CHECK(s.engine.closed && s.vision.closed && s.hub.closed);
}

void test_browser_flag() {
    std::puts("[7] browser opens only when --open and config does not say false");
    std::atomic<int> opened{0};
    std::string last_url;
    std::mutex m;
    auto saved       = g_browser_opener;
    g_browser_opener = [&](const std::string& u) {
        std::lock_guard<std::mutex> g(m);
        last_url = u;
        ++opened;
    };

    {
        Scenario s;
        s.ctx.args.open = true;
        auto t          = raiser({{0.4, SIGINT}});
        run_tail(s.ctx);
        t.join();
        std::this_thread::sleep_for(100ms);
        CHECK(opened == 1);
        std::lock_guard<std::mutex> g(m);
        CHECK(last_url == "http://127.0.0.1:8080/");
    }

    opened = 0;
    {
        Scenario s;
        s.ctx.args.open        = true;
        s.ctx.cfg_open_browser = false;  // config wins (#609)
        auto t                 = raiser({{0.4, SIGINT}});
        run_tail(s.ctx);
        t.join();
        std::this_thread::sleep_for(100ms);
        CHECK(opened == 0);
    }

    opened = 0;
    {
        Scenario s;
        s.ctx.args.open = false;
        auto t          = raiser({{0.4, SIGINT}});
        run_tail(s.ctx);
        t.join();
        std::this_thread::sleep_for(100ms);
        CHECK(opened == 0);
    }

    g_browser_opener = saved;
}

int manual_mode() {
    std::puts("manual mode: opening the browser; press Ctrl+C to stop (twice to end the engine at once)");
    Scenario s;
    s.ctx.args.open             = true;
    s.ctx.args.shutdown_timeout = 5.0;
    s.engine.close_blocks_s     = 3.0;
    return run_tail(s.ctx);
}

}  // namespace

int test_Plot(int argc) {
    if (argc > 1)
        return manual_mode();

    test_clean_ctrl_c();
    test_drain_timeout();
    test_second_ctrl_c();
    test_ready_file_failure();
    test_sigterm();
    test_throwing_closer();
    test_browser_flag();

    std::printf("\n%s (%d failed check%s)\n", g_failures ? "SOME TESTS FAILED" : "ALL TESTS PASSED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}