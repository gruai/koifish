/**
 *  SPDX-FileCopyrightText: 2023-2026 Yingshi Chen <gsp.cys@gmail.com>
 *  SPDX-License-Identifier: MIT
 *
 *  Langevin dynamics = Score(log‑probability)‑driven stochastic gradient flow + Gaussian noise, annealed over time.
 *  Langevin dynamics is the Markov process whose infinitesimal generator is the Fokker–Planck operator corresponding to diffusion in a potential field.
 *  Chat is just sampling process on Langevin dynamics
 *
 *  \brief Langevin dynamics
 *  \author Yingshi Chen
 */

 #include "Transition.hpp"