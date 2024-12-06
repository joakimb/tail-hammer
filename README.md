# quorum_stats

Code for producing the statistics and plots in the paper Consistency-or-Die: Consistency for Key Transparency.</br>
The code uses the gmpy2 package for ultra-high precision arithmetic.

## A note that may save you a lot of time when installing the gmpy2 package
The only tricky part of running this code is to install the gmpy2 package.
If you are on a Linux machine, this is probably not a problem.
But on Windows, this not easy at all.

I used a Windows 11 machine, and a good workaround is to install the Windows Ubuntu App and to install gmpy2 and run the code from there.
Google for assistance if needed, because I do not recall in detail if some magic touch was required for this to work.
Performance is not a problem with the Windows Ubuntu App, guesstimating a 10% penalty compared to running Python code natively.

Installing gmpy2 on Mac can allegedly be a little tricky as well, but not Windows-level insanely-tricky.
