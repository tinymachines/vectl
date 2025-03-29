#ifndef LINUX_CHECK_H
#define LINUX_CHECK_H

#include <cstdlib>

#ifdef __linux__
  #include <linux/version.h>
  #ifndef KERNEL_VERSION
    #define KERNEL_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
  #endif
  #ifndef LINUX_VERSION_CODE
    #error "Linux kernel headers not found or incomplete"
  #endif
#endif

#endif // LINUX_CHECK_H
