"""The code essentially demonstrates how to perform polynomial convolution using the NTT 
   and validates that the NTT-based convolution matches the result obtained using 
   traditional polynomial multiplication and division."""

   # I have already explained in detail the working of this code in the readme file.
   # please refer to it for more details.

import numpy as np

n = 512
q = 12289

def reverse_bits(n):
    b = "{:0{width}b}".format(n, width=9)
    return int(b[::-1], 2)

def generate_positions(n):
    positions = [reverse_bits(x) for x in range(0, n)]
    return positions

def generate_twiddleFactor(n, gamma, q):
    alpha = 1
    tmp = []
    tFactor = []
    inv_tFactor = []

    for x in range(0, n):
        tmp.append(alpha)
        alpha = alpha * gamma % q

    positions = generate_positions(n)

    for x in range(0, n):
        val = tmp[positions[x]]
        inv_val = pow(val, -1, q)
        tFactor.append(val)
        inv_tFactor.append(inv_val)

    return tFactor, inv_tFactor



#NTT transformation using radix-2 Cooley-Tukey method

def cooley_tukey_ntt(a, tFactor, q):
    n = len(a)
    t = n
    m = 1

    while m < n:
        t = t // 2

        for i in range(0, m):
            j1 = 2 * i * t
            j2 = j1 + t - 1
            S = tFactor[m + i]

            for j in range(j1, j2 + 1):
                U = a[j]
                V = a[j + t] * S
                a[j] = (U + V) % q
                a[j + t] = (U - V) % q

        m = 2 * m



#Inverse NTT using the radix-2 Gentleman-Sande method

def gentleman_sande_inv_ntt(a, inv_tFactor, q):
    n = len(a)
    t = 1
    m = n

    while m > 1:
        j1 = 0
        h = m // 2

        for i in range(0, h):
            j2 = j1 + t - 1
            S = inv_tFactor[h + i]

            for j in range(j1, j2 + 1):
                U = a[j]
                V = a[j + t]
                a[j] = (U + V) % q
                a[j + t] = ((U - V) * S) % q

            j1 = j1 + 2 * t

        t = 2 * t
        m = m // 2

    n_inv = pow(n, -1, q)
    for i in range(0, n):
        a[i] = a[i] * n_inv % q

def main():
    positions = generate_positions(n)
    tFactor, inv_tFactor = generate_twiddleFactor(n, 10968, q)

    f = np.zeros(n + 1)
    f[0] = 1
    f[n] = 1

    a = np.random.randint(0, q, n)
    b = np.random.randint(0, q, n)

    #Traditional Approach
    #######
    p = np.remainder(np.polydiv(np.polymul(a[::-1], b[::-1]), f)[1], q).astype(int)[::-1]

    print("Convolution result (p):", p)

    #######

    cooley_tukey_ntt(a, tFactor, q)
    cooley_tukey_ntt(b, tFactor, q)

    #Point Wise Multiplication
    c = np.multiply(a, b)
    
    gentleman_sande_inv_ntt(c, inv_tFactor, q)

    print(c)
    if np.array_equal(c, p):
        print("Your code is CORRECT. Both the Traditional and Fast Approaches are giving the same output.")
    else:
        print("Your code is INCORRECT. There is a Mismatch between the Traditional and Fast approaches.")



if __name__ == "__main__":
    main()
