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

#if EXPERIMENTAL_NULLBUFFER
class NullBuffer : public std::streambuf
{
public:
  int overflow(int c) { return c; }
};

NullBuffer null_buffer;
std::ostream null_stream(&null_buffer);
#endif  // EXPERIMENTAL_NULLBUFFER

#if EASYLOGGING_OFF && EXPERIMENTAL_NULLBUFFER
#define LOG(type)   null_stream << type
#else
#define LOG(type)   std::cout << type
#endif  // defined(EASYLOGGING_OFF)

#define INITIALIZE_EASYLOGGINGPP

#define START_EASYLOGGINGPP(a, b)

#endif // EASYLOGGINGPP_H
