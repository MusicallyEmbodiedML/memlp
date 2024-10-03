// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/* App headers */
#include <xcore/parallel.h>
#include <stdio.h>

#if !defined(FUNCLEARNTEST_C_FN)
#define FUNCLEARNTEST_C_FN __attribute__(( fptrgroup("funclearntest") ))
#endif

/**
 * FuncLearnTest parallel job
 */

DECLARE_JOB(funclearntest_main, (void));
DECLARE_JOB(boo, (void));

void boo(void)
{
    printf("\n---\nThread1:\nBoo!\n---\n");
    for(;;);
}

extern FUNCLEARNTEST_C_FN void funclearntest_main(void);

int main(void)
{
  PAR_JOBS(
    PJOB(funclearntest_main, ()),
    PJOB(boo, ())
    );
}
