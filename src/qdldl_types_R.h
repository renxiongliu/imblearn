#ifndef QDLDL_TYPES_H
# define QDLDL_TYPES_H

# ifdef __cplusplus
extern "C" {
# endif /* ifdef __cplusplus */

// QDLDL integer and float types

typedef long long    QDLDL_int;   /* for indices */
typedef double  QDLDL_float; /* for numerical values  */

// Use bool for logical type if available
#if __STDC_VERSION__ >= 199901L
typedef _Bool QDLDL_bool;
#else
typedef int   QDLDL_bool;
#endif

# ifdef __cplusplus
}
# endif /* ifdef __cplusplus */

#endif /* ifndef QDLDL_TYPES_H */
