//
//  Mockup for easylogging++ that works without an OS.
//
#ifndef EASYLOGGINGPP_H
#define EASYLOGGINGPP_H

#include <iostream>

const std::string INFO = "INFO - ";
const std::string WARNING = "WARN - ";
#ifndef WIN32
const std::string ERROR = "ERROR - ";
#endif

#define LOG(type)   std::cout << type

#define INITIALIZE_EASYLOGGINGPP

#define START_EASYLOGGINGPP(a, b)

#endif // EASYLOGGINGPP_H
