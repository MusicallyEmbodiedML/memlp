# XMOS build commands

## Build app

### xmos_main.cpp

xcc -Wall -std=c++14 -DMLP_VERBOSE -Ideps -report -o XMOS.xe "C:\Users\virgult\Documents\Sussex\XMOS\sw_usb_audio\app_usb_aud_xk_316_mc\src\core\xk-audio-316-mc.xn" src\MLP.cpp src\xmos_main.cpp

### FuncLearnTest

xcc -Wall -std=c++14 -DFUNCLEARN_MAIN -DMLP_VERBOSE -Ideps -report -o XMOS.xe "C:\Users\virgult\Documents\Sussex\XMOS\sw_usb_audio\app_usb_aud_xk_316_mc\src\core\xk-audio-316-mc.xn" src\MLP.cpp src\FuncLearnTest.cpp

### xmos_par_main.c

xcc -c -Wall -std=c++14 -DMLP_VERBOSE -Ideps -report -o XMOS_MLP.o "C:\Users\virgult\Documents\Sussex\XMOS\sw_usb_audio\app_usb_aud_xk_316_mc\src\core\xk-audio-316-mc.xn" src\MLP.cpp

xcc -c -Wall -std=c++14 -DMLP_VERBOSE -Ideps -report -o XMOS_FuncLearnTest.o "C:\Users\virgult\Documents\Sussex\XMOS\sw_usb_audio\app_usb_aud_xk_316_mc\src\core\xk-audio-316-mc.xn" src\FuncLearnTest.cpp

xcc -c -Wall -DMLP_VERBOSE -Ideps -report -o XMOS_c.o "C:\Users\virgult\Documents\Sussex\XMOS\sw_usb_audio\app_usb_aud_xk_316_mc\src\core\xk-audio-316-mc.xn" src\xmos_par_main.c

xcc -o XMOS.xe "C:\Users\virgult\Documents\Sussex\XMOS\sw_usb_audio\app_usb_aud_xk_316_mc\src\core\xk-audio-316-mc.xn" XMOS_MLP.o XMOS_FuncLearnTest.o XMOS_c.o

## Run executable

xrun --io XMOS.xe
