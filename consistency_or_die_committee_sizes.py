from gmpy2 import *
import time
import io

##############################################################
# statistics utilities
##############################################################

# test function for verifying accuracy of the probability computations given the precision settings
# simly computes the sum of the probabilities in the pdf, which should sum close to 1 (depending on precision settings)
def bin_pdf_sum_test(H, p, numIntermediatePrintouts = 10):
  pm1 = mpfr('1') - p
  print("    p = ", p)
  print("1 - p = ", pm1)
  cumulativeH = mpfr('0')
  loopMod = H // numIntermediatePrintouts
  p0 = mpfr('1')
  p1 = pm1 ** H
  b = mpfr('1')
  bMul = H
  bDiv = mpfr('1')
  k = mpz('0')
  while k <= H:
    # compute probability of exactly k successful trials
    #binCoeff = gmpy2.bincoef(H, k)
    #probExactlyK_def = binCoeff * ((p) ** k) * (pm1 ** (H - k))
    #probExactlyK = binCoeff * p0 * p1
    probExactlyK = b * p0 * p1
    cumulativeH += probExactlyK
    if (k % loopMod == 0):
      print("probExactlyK = ", probExactlyK, ", cumulativeH = ", cumulativeH, " at k = ", k)
    p0 *= p
    p1 /= pm1
    # update binomial coefficient
    b *= bMul
    b /= bDiv
    bMul -= 1
    bDiv += 1
    k += 1
  print("After loop, cumulativeH = ", cumulativeH)
  return cumulativeH

def print_mpfr_array(name_str, y):
  print(name_str, " = [{0:.8f}".format(y[0]), sep='', end='')
  for yy in y[1:]:
    print(", {0:.8f}".format(yy), sep='', end='')
  print("]")

# pdf plotting data for visualization
def bin_pdf_plot_data(H, p, k_start, k_end, numIntermediatePrintouts = 100):
  x = []
  y = []
  pm1 = mpfr('1') - p
  cumulativeH = mpfr('0')
  loopMod = (k_end - k_start) // numIntermediatePrintouts
  p0 = mpfr('1')
  p1 = pm1 ** H
  b = mpfr('1')
  bMul = H
  bDiv = mpfr('1')
  k = 0
  while k <= k_end:
    probExactlyK = b * p0 * p1
    cumulativeH += probExactlyK
    if (k >= k_start):
      if (k % loopMod == 0):
        x.append(k)
        y.append(probExactlyK)
    p0 *= p
    p1 /= pm1
    # update binomial coefficient
    b *= bMul
    b /= bDiv
    bMul -= 1
    bDiv += 1
    k += 1
  return x, y

# For Y~Bin(H, p), computes Pr(Y<=k) = sum_{i=0}^{k}[Pr(Y=i)]
def bin_pdf_left_tail_area(H, p, k):
  pm1 = mpfr('1') - p
  cumulativeH = mpfr('0')
  p0 = mpfr('1')
  p1 = pm1 ** H
  b = mpfr('1')
  bMul = H
  bDiv = mpfr('1')
  i = mpz('0')
  while i <= k:
    probExactlyK = b * p0 * p1
    cumulativeH += probExactlyK
    p0 *= p
    p1 /= pm1
    # update binomial coefficient
    b *= bMul
    b /= bDiv
    bMul -= 1
    bDiv += 1
    i += 1
  return cumulativeH


##############################################################
# Background on committees, committee selection and committee sizes.
#
# Consider a committee selection procedure where users are selected
# for committee duty independently and at random with probability p.
# You may think of the process as each user computing a Verifiable Random
# Function (VRF) using its own private key together with some public seed
# as input, and passing as a committee member if the output value is below
# a target value (to match with probability p).
# The VRF allows any other user or system owner to verify that the computation
# (and the claim to the seat on the committee) is indeed correct, so that
# we can avoid trusting malicious entities that claim to be on the committee
# when they are not.
#
# If we have a communication channel *with* guaranteed output delivery (GOD), so that no
# entity can suppress output, then it is sufficient to have an honest majority
# in the committee for the final committee verdict/computation to be the honest
# and correct one. In this case we need to choose a committee of a sufficiently
# large size so that we have an honest majority with very high probability (whp).
#
# If we have a communication channel *without* guaranteed output delivery, so that,
# say, the system owner can suppress output, then it is not sufficient to have a
# mere honest majority, since all honest committee members can be suppressed at will.
# In this case we need to proceed differently to ensure an honest committee end-result.
# We first choose a large enough committee size. Then we also set a threshold value T,
# such that
# * there are fewer than T malicious users on the committee whp, and
# * there are more than T honest users on the committee whp.
# Any suggestion/computation that wins at least T committee member votes wins.
#
# The functions below detail the statistical analysis and compute optimal committee
# sizes for the case *without* GOD.
# We use the following notation.
# U: the total number of users in the system
# H: the number of (actively) honest users
# M: the number of (actively) malicious users
# S: the number of silent/non-responsive users
# So we have U = H + M + S.
# For committee selection, each user in U computes its VRF output and checks if they
# are on the committee. These are independent Bernoulli trials with probability p.
# In the analysis below, we consider the sum of those independent trials over the
# H and M honest and malicious users separately.
##############################################################

##############################################################
# Committee sizes *without* guaranteed output delivery
# Consistency-or-Die (cod) - style
##############################################################

# X~Bin(M, p), num malicious committee members
# Y~Bin(H, p), num honest committee members
# compares cut off values k for simultaneously "minimizing"
#    the right tail R of X, Pr(X>=k), and
#    the left tail L of Y, Pr(Y<k).
# In practice we need both L and R to be small enough, but it is application dependent what that actually means.
def cod_committee_cutoff_probabilities(H, M, p):
  if (M >= H):
    return -1, -1, -1
  expectedCommitteeSizeH = mpz(p*H) # we need not iterate k any further than this
  if (expectedCommitteeSizeH > M):
    return -2, -2, -2
  expectedCommitteeSizeM = mpz(p*M)
  pm1 = mpfr('1') - p
  cumulativeH = mpfr('0')
  cumulativeM_opposite = mpfr('1')
  p0 = mpfr('1')
  p1H = pm1 ** H
  bH = mpfr('1')
  bMulH = H
  bDiv = mpfr('1')
  p1M = pm1 ** M
  bM = mpfr('1')
  bMulM = M
  k = 0
  while k <= expectedCommitteeSizeH:
    probExactlyK_H = bH * p0 * p1H
    cumulativeH += probExactlyK_H
    probExactlyK_M = bM * p0 * p1M
    cumulativeM_opposite -= probExactlyK_M
    if (cumulativeH > cumulativeM_opposite):
      return k, cumulativeH, cumulativeM_opposite
    p0 *= p
    p1H /= pm1
    p1M /= pm1
    # update binomial coefficients
    bH *= bMulH
    bH /= bDiv
    bM *= bMulM
    bM /= bDiv
    bMulH -= 1
    bMulM -= 1
    bDiv += 1
    k += 1
  return -3, -3, -3

def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents

def cod_cutoffs_batch(populationSize, Hpercentage, Mpercentage, startCommitteeSize, endCommitteeSize, stepSize):
  U = mpz(populationSize)
  H = U * Hpercentage // 100
  M = U * Mpercentage // 100
  S = U - H - M
  fileName = "cod_cutoffs_H%d_M%d.txt" % (Hpercentage, Mpercentage)
  print("writing to file ", fileName, sep='')
  x_expected_committee_size = []
  x_cutoffs = []
  y = []
  with open(fileName, "w") as f:
    f.write("U = %d (total num users) = 2^{%.2f}\n" % (U, gmpy2.log2(U)))
    f.write("H = %d (num actively honest users, %d%%)\n" % (H, Hpercentage))
    f.write("M = %d (num actively malicious users, %d%%)\n" % (M, Mpercentage))
    f.write("S = %d (num silent/non-responsive users, %d%%)\n" % (S, 100 - Hpercentage - Mpercentage))
    M += H // 2
    f.write("Metric: given X~Bin(M + (H/2), p), Y~Bin(H, p), optimal k s.t. Pr(X>=k) = Pr(Y<=k)\n\n")
    for expectedCommitteeSize in range(startCommitteeSize, endCommitteeSize+1, stepSize):
      p = mpfr('1') * expectedCommitteeSize / U
      f.write(print_to_string("p = ", p, " (expected committee size = ", expectedCommitteeSize, ")", sep=''))
      k, L, R = cod_committee_cutoff_probabilities(H, M, p)
      f.write("  k = %d\n" % (k))
      logL = gmpy2.log2(L)
      logR = gmpy2.log2(R)
      f.write(print_to_string("  log(L) = ", logL, sep=''))
      f.write(print_to_string("  log(R) = ", logR, sep=''))
      f.write("\n")
      x_expected_committee_size.append(expectedCommitteeSize)
      x_cutoffs.append(k)
      y.append(logL) # the larger of the two (pessimistic)
    # summary printing for plotting convenience
    f.write(print_to_string("x_expected_committee_sizes = ", x_expected_committee_size, sep=''))
    f.write(print_to_string("x_cutoffs = ", x_cutoffs, sep=''))
    f.write("y = [{0:.2f}".format(y[0]))
    for yy in y[1:]:
      f.write(print_to_string(", {0:.2f}".format(yy), sep='', end=''))
    f.write("] # log of probability of failure (log of left H tail area = log of right M tail area)\n")

def pdf_visualization(populationSize, Hpercentage, Mpercentage, p, k_start, k_end, numIntermediatePrintouts):
  U = mpfr(populationSize)
  H = U * Hpercentage // 100
  M = U * Mpercentage // 100
  A = M + (H // 2)
  x_M, y_M = bin_pdf_plot_data(A, p, k_start, k_end, numIntermediatePrintouts)
  print("x = ", x_M, sep='')
  print_mpfr_array("y_M", y_M)
  x_H, y_H = bin_pdf_plot_data(H, p, k_start, k_end, numIntermediatePrintouts)
  print_mpfr_array("y_H", y_H)


if __name__ == '__main__':
  precision = 1000
  ctx = get_context()
  print("old precision = ", ctx.precision)
  ctx.precision = precision
  print("new precision = ", ctx.precision)
  #cod_cutoffs_batch(2**30, 79, 1, 100, 7000, 1)
  #cod_cutoffs_batch(2**30, 75, 5, 100, 20000, 1)
  #cod_cutoffs_batch(2**30, 70, 10, 100, 20000, 1)
  #cod_cutoffs_batch(2**30, 65, 15, 100, 26000, 1)
  #cod_cutoffs_batch(2**30, 60, 20, 7900, 8000, 1) # 30-bit
  #cod_cutoffs_batch(2**30, 60, 20, 22900, 23000, 1) # 80-bit
  #cod_cutoffs_batch(2**30, 60, 20, 37400, 37500, 1) # 128-bit
  #cod_cutoffs_batch(2**30, 60, 20, 76200, 76300, 1) # 256-bit
  #cod_cutoffs_batch(2**30, 60, 20, 100, 76300, 100) # sparse for plotting
  #cod_cutoffs_batch(2**30, 55, 25, 1193000, 1193000, 1)
  #pdf_visualization(2**30, 60, 20, mpfr('0.00001'), 4000, 8000, 400)# higher p
  #pdf_visualization(2**30, 60, 20, mpfr('0.000001'), 400, 800, 400)# lower p
  pdf_visualization(2**30, 60, 20, mpfr('0.0000005'), 150, 450, 299)# even lower p
