/**
 * More https://skebanga.github.io/embedded-python-pybind11/
 */
#ifdef _USE_WANDB_
#include <pybind11/embed.h>  // everything needed for embedding

#include <iostream>
namespace py = pybind11;
void _WANDB_int(std::string &proj, int flag = 0x0) {
    py::scoped_interpreter guard{};
    auto locals      = py::dict();
    std::string path = "/root/X/lic/src/py-models/batch_wandb.py";
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Could not open file " << path << std::endl;
        return 1;
    }

    std::string str((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    py::exec(str);
}

void _WANDB_log(double a) {
    main_11();

    py::scoped_interpreter guard{};  // start the interpreter and keep it alive
    // py::print("Hello, World!");  // use the Python API A
    // py::object sys = py::module::import("sys");
    // std::string sys_version = sys.attr("version").cast<std::string>();
    // std::cout << "sys version: " << sys_version << std::endl;
    auto locals = py::dict();
    py::exec(R"(
            import scipy
            import wandb
            print("hello workd!!!")

            wandb.log({'acc':acc},step=epoch)
        )",
             py::globals(), locals);

    // py::object scipy = py::module::import("scipy");
    // std::string scipy_version = scipy.attr("__version__").cast<std::string>();
    // std::cout << "scipy version: " << scipy_version << std::endl;
}
#endif