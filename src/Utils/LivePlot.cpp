/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
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