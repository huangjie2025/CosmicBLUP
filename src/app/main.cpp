#include "blup_app.h"
#include "version.h"
#include <iostream>

using namespace cosmic;

int main(int argc, char** argv) {
    try {
        if (argc > 1) {
            Config cfg = parse_solver_cli(argc, argv);
            if (cfg.show_version) {
                std::cout << "CosmicBLUP v" << COSMICBLUP_VERSION << std::endl;
                return 0;
            }
            std::cout << "CosmicBLUP Starting..." << std::endl;
            SolverApp app(cfg);
            app.run();
        } else {
            std::cerr << "Usage: cosmicblup [options]\n";
            std::cerr << "Try `cosmicblup --help` or `cosmicblup --help relationship`.\n";
            std::cerr << "See docs/USER_GUIDE_CN.md or docs/USER_GUIDE_EN.md for examples.\n";
            return 1;
        }
    } catch (std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
