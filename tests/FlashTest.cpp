#include "UnitTest.hpp"

# if defined(__XS3A__)  // Only works on XMOS (obviously)


#include "utils/Flash.hpp"
#include "easylogging++.h"

#include <vector>
#include <algorithm>
#include <cstdint>
#include <cstdio>

UNIT(FlashReadWrite) {

    const std::string payload_str("The quick brown fox jumped over the lazy dog.");
    std::vector<uint8_t> exp_payload(payload_str.begin(), payload_str.end());
    exp_payload.push_back('\0');

    gFlash.SetPayload(exp_payload);
    gFlash.connect();
    gFlash.PrintFlashInfo();
    gFlash.WriteToFlash();
    gFlash.SetPayload({});
    gFlash.ReadFromFlash();

    std::vector<uint8_t> payload;
    gFlash.GetPayload(payload);
    std::string read_str(reinterpret_cast<char *>(payload.data()), payload.size());
    LOG(INFO) << "Expected: " << payload_str << std::endl;
    LOG(INFO) << "Actual: " << read_str << std::endl;
    ASSERT_TRUE(payload.size() == exp_payload.size());
    ASSERT_TRUE(payload == exp_payload);
}


#if 0
int main(int argc, char* argv[]) {
    START_EASYLOGGINGPP(argc, argv);
    microunit::UnitTester::Run();
    return 0;
}
#endif

#endif  // __XS3A__