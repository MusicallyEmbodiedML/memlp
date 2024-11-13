#include "microunit.h"
#include "easylogging++.h"

INITIALIZE_EASYLOGGINGPP

// Include all tests (yes, in CPP) here!

//#include "IrisDatasetTest.cpp"
//#include "LayerTest.cpp"
//#include "LossTest.cpp"
#include "MLPTest.cpp"
//#include "NodeTest.cpp"
//#include "FlashTest.cpp"
//#include "SerialiseTest.cpp"

int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}
