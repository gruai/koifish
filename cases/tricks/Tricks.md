It's really hard to train 10B models on single GPU. Koifish have tried many tricks and am attempting even more.
The following table lists some tricks and its working status.

| Tricks | Maturity  | Effectiveness |Todo
|:-------------:|:--------------:|:--------------:|:--------------:|
| Bit weight         | 🧪           | ★ ★ ★ ☆ ☆        ||
| Rope    | 🧪      | ★ ★ ☆ ☆ ☆     ||
| Rematerialisation    | ✔️      | ★ ★ ★ ★ ☆     ||
| Weight Tying    | ✔️      | ★ ★ ★ ☆ ☆     ||
| Hybrid Adam    | ✔️      | ★ ★ ★ ☆ ☆     ||
| Lion         | 🔒           | ★ ★ ☆ ☆ ☆        ||
| Subsampling    | 🌿      |  ★ ☆ ☆ ☆ ☆    ||
| LORA    |    🧪   | ★ ★ ☆ ☆ ☆     ||
| Mixture of models    |    🌿   | ★ ★ ☆ ☆ ☆     ||
| Sparse model    |    🌱   |     ||

Note 1.  The meaning of symbols for maturity 
1. 🌱 Prototype      
2. 🌿 Alpha (Buggy)  
3. 🧪 Beta (Functional but rough)  
4. 🌳 Stable  
5. ✔️ Production
6. 🏆 Wonderful
7. 🔒 Finished & Locked 
8. 🏛 Legacy (Deprecated)

