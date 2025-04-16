#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/arrayobject.h"
#include "qpSWIFT.h"

#if PY_MAJOR_VERSION >= 3
#define qpLong_check PyLong_Check
#define qp_getlong PyLong_AsLong
#else
#define qpLong_check PyInt_Check
#define qp_getlong PyInt_AsLong
#endif

qp_real *getptr(PyArrayObject *ar)
{
    // qp_real *m = (qp_real *)PyArray_DATA(PyArray_GETCONTIGUOUS(ar));
    //  return m;

    // PyArrayObject *tmp_arr;

    // tmp_arr = PyArray_GETCONTIGUOUS(ar);
    // qp_real *m = (qp_real *)PyArray_DATA(tmp_arr);
    // //Py_DECREF(tmp_arr);
    // return m;

    PyArrayObject *tmp_arr;
    PyArrayObject *new_owner;
    tmp_arr = PyArray_GETCONTIGUOUS(ar);
    new_owner = (PyArrayObject *)PyArray_Cast(tmp_arr, NPY_DOUBLE);
    Py_DECREF(tmp_arr);
    // qp_real *ptr = PyArray_DATA(new_owner);
    //   Py_DECREF(new_owner);
    return new_owner;
}

static PyObject *method_qpSWIFT(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyArrayObject *c, *h, *b = NULL;
    PyArrayObject *P;
    PyArrayObject *A = NULL;
    PyArrayObject *G;

    PyObject *opts = NULL;
    PyObject *opts_maxiter = NULL, *opts_abstol = NULL, *opts_reltol = NULL, *opts_sigma = NULL, *opts_verbose = NULL, *opts_output = NULL;

    qp_int opts_output_level = 10;

    qp_real *Ppr, *Apr = NULL, *Gpr;
    qp_real *cpr, *hpr, *bpr = NULL;

    npy_intp *dims;

    PyObject *basic_info = NULL, *adv_info = NULL, *result;

    /* Temporary Pointer */
    qp_real *temptr;

    PyArrayObject *ctemp, *htemp, *btemp = NULL;
    PyArrayObject *Ptemp, *Gtemp, *Atemp = NULL;

    /* Results Pointer */
    PyArrayObject *sol_x, *sol_y = NULL, *sol_z = NULL, *sol_s = NULL;
    /* Results Pointer */

    /* Results Dimensions */
    npy_intp sol_xdim[1], sol_ydim[1], sol_zdim[1], sol_sdim[1];
    /* Results Dimensions */

    static char *kwlist[] = {"c", "h", "P", "G", "A", "b", "opts", NULL};

    static char *argparse_string = "O!O!O!O!|O!O!O!";

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, argparse_string, kwlist,
                                     &PyArray_Type, &c,
                                     &PyArray_Type, &h,
                                     &PyArray_Type, &P,
                                     &PyArray_Type, &G,
                                     &PyArray_Type, &A,
                                     &PyArray_Type, &b,
                                     &PyDict_Type, &opts))

    {
        return NULL;
    }

    qp_int n, m, p;
    dims = PyArray_DIMS(c);
    n = (qp_int)dims[0];
    dims = PyArray_DIMS(h);
    m = (qp_int)dims[0];
    if (b && A)
    {
        dims = PyArray_DIMS(b);
        p = (qp_int)dims[0];
    }
    else
    {
        p = 0;
    }

    /*Check Input Data here*/
    /*---- c vector ------*/
    if (!PyArray_ISFLOAT(c) || (qp_int)(PyArray_NDIM(c) != 1))
    {
        PyErr_SetString(PyExc_TypeError, "c must be a floating array with one dimension");
        return NULL;
    }
    /*---- c vector ------*/

    /*---- h vector ------*/
    if (!PyArray_ISFLOAT(h) || (qp_int)(PyArray_NDIM(h) != 1))
    {
        PyErr_SetString(PyExc_TypeError, "h must be a floating array with one dimension");
        return NULL;
    }
    /*---- h vector ------*/

    /*---- b vector and A Matrix ------*/
    if (b && A)
    {
        /*---- b vector ------*/
        if (!PyArray_ISFLOAT(b) || (qp_int)(PyArray_NDIM(b) != 1))
        {
            PyErr_SetString(PyExc_TypeError, "b must be a floating array with one dimension");
            return NULL;
        }
        /*---- b vector ------*/

        /*---- A Matrix ------*/
        if (!PyArray_ISFLOAT(A) || (qp_int)(PyArray_NDIM(A) != 2))
        {
            PyErr_SetString(PyExc_TypeError, "A must be a floating matrix with two dimensions");
            return NULL;
        }

        if (((qp_int)PyArray_DIM(A, 0) != p) || (qp_int)(PyArray_DIM(A, 1) != n))
        {
            PyErr_SetString(PyExc_TypeError, "b and A do not have compatible dimensions");
            return NULL;
        }
        /*---- A Matrix ------*/
    }

    /*---- G Matrix ------*/
    if (!PyArray_ISFLOAT(G) || (qp_int)(PyArray_NDIM(G) != 2))
    {
        PyErr_SetString(PyExc_TypeError, "G must be a floating matrix with two dimensions");
        return NULL;
    }

    if ((qp_int)(PyArray_DIM(G, 0) != m) || (qp_int)(PyArray_DIM(G, 1) != n))
    {
        PyErr_SetString(PyExc_TypeError, "h and G do not have compatible dimensions");
        return NULL;
    }
    /*---- G Matrix ------*/

    /*---- P Matrix ------*/
    if (!PyArray_ISFLOAT(P) || (qp_int)(PyArray_NDIM(P) != 2))
    {
        PyErr_SetString(PyExc_TypeError, "P must be a floating matrix with two dimensions");
        return NULL;
    }

    if ((qp_int)(PyArray_DIM(P, 0) != n) || (qp_int)(PyArray_DIM(P, 1) != n))
    {
        PyErr_SetString(PyExc_TypeError, "c and P do not have compatible dimensions");
        return NULL;
    }
    /*---- P Matrix ------*/

    /*----- options ------*/

    /*---Initialize default options ----*/
    settings inopts;
    inopts.abstol = ABSTOL;
    inopts.reltol = RELTOL;
    inopts.maxit = MAXIT;
    inopts.sigma = SIGMA;
    inopts.verbose = VERBOSE;
    /*---Initialize default options ----*/

    if (opts)
    {
        opts_maxiter = PyDict_GetItemString(opts, "MAXITER");

        if (opts_maxiter)
        {
            Py_INCREF(opts_maxiter);
            if (qpLong_check(opts_maxiter) && qp_getlong(opts_maxiter) < 200 && qp_getlong(opts_maxiter) > 0)
            {
                inopts.maxit = (qp_int)qp_getlong(opts_maxiter);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "max iterations must be between 0 and 200, using the default value of 100");
                return NULL;
            }
            Py_DECREF(opts_maxiter);
        }

        opts_abstol = PyDict_GetItemString(opts, "ABSTOL");

        if (opts_abstol)
        {
            Py_INCREF(opts_abstol);
            if (PyFloat_Check(opts_abstol) && PyFloat_AsDouble(opts_abstol) < 1.0 && PyFloat_AsDouble(opts_abstol) > 0.0)
            {
                inopts.abstol = (qp_real)PyFloat_AsDouble(opts_abstol);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "absolute tolerance must be between 0 and 1, using the default value");
                return NULL;
            }
            Py_DECREF(opts_abstol);
        }

        opts_reltol = PyDict_GetItemString(opts, "RELTOL");

        if (opts_reltol)
        {
            Py_INCREF(opts_reltol);
            if (PyFloat_Check(opts_reltol) && PyFloat_AsDouble(opts_reltol) < 1.0 && PyFloat_AsDouble(opts_reltol) > 0.0)
            {
                inopts.reltol = (qp_real)PyFloat_AsDouble(opts_reltol);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "realtive tolerance must be between 0 and 1, using the default value");
                return NULL;
            }
            Py_DECREF(opts_reltol);
        }

        opts_sigma = PyDict_GetItemString(opts, "SIGMA");

        if (opts_sigma)
        {
            Py_INCREF(opts_sigma);
            if (PyFloat_Check(opts_sigma) && PyFloat_AsDouble(opts_sigma) > 0.0)
            {
                inopts.sigma = (qp_real)PyFloat_AsDouble(opts_sigma);
            }
            else
            {
                PyErr_SetString(PyExc_TypeError, "sigma must be positive, using the default value");
                return NULL;
            }
            Py_DECREF(opts_sigma);
        }

        opts_verbose = PyDict_GetItemString(opts, "VERBOSE");

        if (opts_verbose)
        {
            Py_INCREF(opts_verbose);
            if (qpLong_check(opts_verbose) && qp_getlong(opts_verbose) >= 0)
            {
                inopts.verbose = (qp_int)qp_getlong(opts_verbose);
            }
            else
            {
                //  PyErr_WarnEx(PyExc_UserWarning, "verbose must be non-negative, using the default value", 1);
                PyErr_SetString(PyExc_TypeError, "verbose must be non-negative integer, using the default value");
                return NULL;
            }
            Py_DECREF(opts_verbose);
        }

        opts_output = PyDict_GetItemString(opts, "OUTPUT");

        if (opts_output)
        {
            Py_INCREF(opts_output);
            if (qpLong_check(opts_output) && qp_getlong(opts_output) >= 0)
            {
                opts_output_level = (qp_int)qp_getlong(opts_output);
            }
            else
            {
                PyErr_WarnEx(PyExc_UserWarning, "output must be non-negative, using the default value", 1);
                // PyErr_SetString(PyExc_TypeError, "verbose must be non-negative integer, using the default value");
                // return NULL;
            }
            Py_DECREF(opts_output);
        }
    }

    /*----- options ------*/

    /*Check Input Data here*/

    /*** Get Data Pointers ***/
    ctemp = getptr(c);
    cpr = (qp_real *)PyArray_DATA(ctemp);

    htemp = getptr(h);
    hpr = (qp_real *)PyArray_DATA(htemp);
    if (b && A)
    {
        btemp = getptr(b);
        bpr = (qp_real *)PyArray_DATA(btemp);
        Atemp = getptr(A);
        Apr = (qp_real *)PyArray_DATA(Atemp);
    }
    Ptemp = getptr(P);
    Ppr = (qp_real *)PyArray_DATA(Ptemp);
    Gtemp = getptr(G);
    Gpr = (qp_real *)PyArray_DATA(Gtemp);
    /*** Get Data Pointers ***/

    QP *myQP;

    myQP = QP_SETUP_dense(n, m, p, Ppr, Apr, Gpr, cpr, hpr, bpr, NULL, ROW_MAJOR_ORDERING);

    /*---- Copy Settings if any---*/
    myQP->options->abstol = inopts.abstol;
    myQP->options->reltol = inopts.reltol;
    myQP->options->maxit = inopts.maxit;
    myQP->options->sigma = inopts.sigma;
    myQP->options->verbose = inopts.verbose;

    /*---- Copy Settings if any---*/

    if (myQP->options->verbose > 0)
    {
        PRINT("\n****qpSWIFT : Sparse Quadratic Programming Solver****\n\n");
        PRINT("================Settings Applied======================\n");
        PRINT("Maximum Iterations : %ld \n", myQP->options->maxit);
        PRINT("ABSTOL             : %e \n", myQP->options->abstol);
        PRINT("RELTOL             : %e \n", myQP->options->reltol);
        PRINT("SIGMA              : %e \n", myQP->options->sigma);
        PRINT("VERBOSE            : %ld \n", myQP->options->verbose);
        PRINT("Permutation vector : AMD Solver\n\n");
    }

    qp_int ExitCode;
    Py_BEGIN_ALLOW_THREADS;
    ExitCode = QP_SOLVE(myQP);
    Py_END_ALLOW_THREADS;

    sol_xdim[0] = myQP->n;
    sol_x = (PyArrayObject *)PyArray_SimpleNew(1, sol_xdim, NPY_DOUBLE);
    temptr = PyArray_DATA(sol_x);
    for (qp_int i = 0; i < myQP->n; ++i)
    {
        temptr[i] = myQP->x[i];
    }

    switch (opts_output_level)
    {
    case 1:
        basic_info = Py_BuildValue("{s:l,s:l,s:d,s:d}",
                                   "ExitFlag", ExitCode,
                                   "Iterations", myQP->stats->IterationCount,
                                   "Setup_Time", myQP->stats->tsetup,
                                   "Solve_Time", myQP->stats->tsolve);

        result = Py_BuildValue("{s:O,s:O}",
                               "sol", sol_x,
                               "basicInfo", basic_info);

        Py_DECREF(basic_info);
        break;
    case 2:
        basic_info = Py_BuildValue("{s:l,s:l,s:d,s:d}",
                                   "ExitFlag", ExitCode,
                                   "Iterations", myQP->stats->IterationCount,
                                   "Setup_Time", myQP->stats->tsetup,
                                   "Solve_Time", myQP->stats->tsolve);

        if (b && A)
        {
            sol_ydim[0] = myQP->p;
            sol_y = (PyArrayObject *)PyArray_SimpleNew(1, sol_ydim, NPY_DOUBLE);
            temptr = PyArray_DATA(sol_y);
            for (qp_int i = 0; i < myQP->p; ++i)
            {
                temptr[i] = myQP->y[i];
            }
        }

        sol_zdim[0] = myQP->m;
        sol_z = (PyArrayObject *)PyArray_SimpleNew(1, sol_zdim, NPY_DOUBLE);
        temptr = PyArray_DATA(sol_z);
        for (qp_int i = 0; i < myQP->m; ++i)
        {
            temptr[i] = myQP->z[i];
        }

        sol_sdim[0] = myQP->m;
        sol_s = (PyArrayObject *)PyArray_SimpleNew(1, sol_sdim, NPY_DOUBLE);
        temptr = PyArray_DATA(sol_s);
        for (qp_int i = 0; i < myQP->m; ++i)
        {
            temptr[i] = myQP->s[i];
        }

        if (b && A)
        {
            adv_info = Py_BuildValue("{s:d,s:d,s:d,s:O,s:O,s:O}",
                                     "fval", myQP->stats->fval,
                                     "kktTime", myQP->stats->kkt_time,
                                     "ldlTime", myQP->stats->ldl_numeric,
                                     "y", sol_y,
                                     "z", sol_z,
                                     "s", sol_s);
        }
        else
        {
            adv_info = Py_BuildValue("{s:d,s:d,s:d,s:O,s:O}",
                                     "fval", myQP->stats->fval,
                                     "kktTime", myQP->stats->kkt_time,
                                     "ldlTime", myQP->stats->ldl_numeric,
                                     "z", sol_z,
                                     "s", sol_s);
        }
        result = Py_BuildValue("{s:O,s:O,s:O}",
                               "sol", sol_x,
                               "basicInfo", basic_info,
                               "advInfo", adv_info);

        Py_DECREF(basic_info);

        Py_DECREF(adv_info);

        if (sol_y)
        {
            Py_DECREF(sol_y);
        }
        Py_DECREF(sol_z);
        Py_DECREF(sol_s);

        break;
    default:
        result = Py_BuildValue("{s:O}",
                               "sol", sol_x);

        break;
    }

    Py_DECREF(sol_x);

    Py_DECREF(Ptemp);
    Py_DECREF(ctemp);
    Py_DECREF(Gtemp);
    Py_DECREF(htemp);
    if (Atemp)
    {
        Py_DECREF(Atemp);
    }

    if (btemp)
    {
        Py_DECREF(btemp);
    }

    QP_CLEANUP_dense(myQP);

    return result;
}

static PyObject *method_qp_SWIFT_sparse(PyObject *self, PyObject *args, PyObject *kwargs)
{
    /* Input arrays and objects */
    PyArrayObject *c, *h, *b = NULL;
    PyObject *P, *G, *A = NULL;
    PyObject *opts = NULL;

    /* For constructing the CSC representation for P */
    PyArrayObject *P_data_arr, *P_indptr_arr_new, *P_indices_arr_new;

    /* For G extraction (expects a scipy.sparse.csc_matrix) */
    PyObject *G_data_obj, *G_indices_obj, *G_indptr_obj, *G_shape_obj;
    PyArrayObject *G_data_arr, *G_indices_arr, *G_indptr_arr;

    /* For A extraction if provided */
    PyObject *A_data_obj = NULL, *A_indices_obj = NULL, *A_indptr_obj = NULL, *A_shape_obj = NULL;
    PyArrayObject *A_data_arr = NULL, *A_indices_arr = NULL, *A_indptr_arr = NULL;

    /* Pointers to data */
    qp_real *cpr, *hpr, *bpr = NULL;
    qp_real *Ppr, *Gpr, *Apr = NULL;
    qp_int *Pjc, *Pir;
    qp_int *Gjc, *Gir;
    qp_int *Ajc = NULL, *Air = NULL;

    PyObject *basic_info = NULL, *adv_info = NULL, *result;

    /* Results arrays */
    PyArrayObject *sol_x, *sol_y = NULL, *sol_z = NULL, *sol_s = NULL;
    npy_intp sol_xdim[1], sol_ydim[1], sol_zdim[1], sol_sdim[1];

    /* Options */
    PyObject *opts_maxiter = NULL, *opts_abstol = NULL, *opts_reltol = NULL;
    PyObject *opts_sigma = NULL, *opts_verbose = NULL, *opts_output = NULL;
    qp_int opts_output_level = 10;

    /* Dimensions and temporary variables */
    qp_int n, m, p = 0;
    qp_int G_nrows, G_ncols;
    qp_int A_nrows = 0, A_ncols = 0;

    static char *kwlist[] = {"c", "h", "P", "G", "A", "b", "opts", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!OO|OOO", kwlist,
                                     &PyArray_Type, &c,
                                     &PyArray_Type, &h,
                                     &P,    // now just an array (diagonal)
                                     &G,
                                     &A,
                                     &b,
                                     &opts))
    {
        return NULL;
    }

    /* Validate c */
    if (!PyArray_ISFLOAT(c) || (PyArray_NDIM(c) != 1)) {
        PyErr_SetString(PyExc_TypeError, "c must be a floating array with one dimension");
        return NULL;
    }
    n = (qp_int)PyArray_DIM(c, 0);

    /* Validate h */
    if (!PyArray_ISFLOAT(h) || (PyArray_NDIM(h) != 1)) {
        PyErr_SetString(PyExc_TypeError, "h must be a floating array with one dimension");
        return NULL;
    }
    m = (qp_int)PyArray_DIM(h, 0);

    /* Check b and A */
    if (b && A) {
        if (!PyArray_ISFLOAT(b) || (PyArray_NDIM(b) != 1)) {
            PyErr_SetString(PyExc_TypeError, "b must be a floating array with one dimension");
            return NULL;
        }
        p = (qp_int)PyArray_DIM(b, 0);
    }

    /* --- Process P as a 1D array holding the diagonal entries --- */
    if (!PyArray_Check(P) || !PyArray_ISFLOAT(P) || (PyArray_NDIM(P) != 1)) {
         PyErr_SetString(PyExc_TypeError, "P must be a floating array with one dimension");
         return NULL;
    }
    if (PyArray_DIM(P, 0) != n) {
         PyErr_SetString(PyExc_ValueError, "Length of P must equal length of c");
         return NULL;
    }
    P_data_arr = (PyArrayObject *)PyArray_FROMANY(P, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!P_data_arr) {
         PyErr_SetString(PyExc_TypeError, "Error converting P to a numpy array");
         return NULL;
    }
    Ppr = (qp_real *)PyArray_DATA(P_data_arr);

    /* Build the CSC representation for the diagonal matrix:
       - P_indptr: [0, 1, 2, ..., n]
       - P_indices: [0, 1, 2, ..., n-1] */
    {
         npy_intp dims_indptr[1] = {n + 1};
         P_indptr_arr_new = (PyArrayObject *)PyArray_SimpleNew(1, dims_indptr, NPY_INT64);
         if (!P_indptr_arr_new) {
              Py_DECREF(P_data_arr);
              PyErr_SetString(PyExc_MemoryError, "Could not allocate P_indptr array");
              return NULL;
         }
         Pjc = (qp_int *)PyArray_DATA(P_indptr_arr_new);
         for (qp_int j = 0; j <= n; j++) {
              Pjc[j] = j;
         }
    }
    {
         npy_intp dims_indices[1] = {n};
         P_indices_arr_new = (PyArrayObject *)PyArray_SimpleNew(1, dims_indices, NPY_INT64);
         if (!P_indices_arr_new) {
              Py_DECREF(P_data_arr);
              Py_DECREF(P_indptr_arr_new);
              PyErr_SetString(PyExc_MemoryError, "Could not allocate P_indices array");
              return NULL;
         }
         Pir = (qp_int *)PyArray_DATA(P_indices_arr_new);
         for (qp_int j = 0; j < n; j++) {
              Pir[j] = j;
         }
    }
    /* --- End P processing --- */

    /* --- Process G as a CSC matrix --- */
    G_data_obj = PyObject_GetAttrString(G, "data");
    G_indices_obj = PyObject_GetAttrString(G, "indices");
    G_indptr_obj = PyObject_GetAttrString(G, "indptr");
    G_shape_obj = PyObject_GetAttrString(G, "shape");

    if (!G_data_obj || !G_indices_obj || !G_indptr_obj || !G_shape_obj) {
        PyErr_SetString(PyExc_TypeError, "G must be a scipy.sparse.csc_matrix with data, indices, indptr, shape attributes");
        return NULL;
    }
    G_data_arr = (PyArrayObject *)PyArray_FROMANY(G_data_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    G_indices_arr = (PyArrayObject *)PyArray_FROMANY(G_indices_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    G_indptr_arr = (PyArrayObject *)PyArray_FROMANY(G_indptr_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);

    if (PyTuple_Check(G_shape_obj)) {
        G_nrows = (qp_int)PyLong_AsLong(PyTuple_GetItem(G_shape_obj, 0));
        G_ncols = (qp_int)PyLong_AsLong(PyTuple_GetItem(G_shape_obj, 1));
    } else {
        PyErr_SetString(PyExc_TypeError, "G.shape is not a tuple");
        return NULL;
    }
    if (G_nrows != m || G_ncols != n) {
        PyErr_SetString(PyExc_TypeError, "G must have compatible dimensions with h and c");
        return NULL;
    }

    /* --- Process A if provided --- */
    if (A && b) {
        A_data_obj = PyObject_GetAttrString(A, "data");
        A_indices_obj = PyObject_GetAttrString(A, "indices");
        A_indptr_obj = PyObject_GetAttrString(A, "indptr");
        A_shape_obj = PyObject_GetAttrString(A, "shape");

        if (!A_data_obj || !A_indices_obj || !A_indptr_obj || !A_shape_obj) {
            PyErr_SetString(PyExc_TypeError, "A must be a scipy.sparse.csc_matrix with data, indices, indptr, shape attributes");
            return NULL;
        }
        A_data_arr = (PyArrayObject *)PyArray_FROMANY(A_data_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        A_indices_arr = (PyArrayObject *)PyArray_FROMANY(A_indices_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
        A_indptr_arr = (PyArrayObject *)PyArray_FROMANY(A_indptr_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);

        if (PyTuple_Check(A_shape_obj)) {
            A_nrows = (qp_int)PyLong_AsLong(PyTuple_GetItem(A_shape_obj, 0));
            A_ncols = (qp_int)PyLong_AsLong(PyTuple_GetItem(A_shape_obj, 1));
        } else {
            PyErr_SetString(PyExc_TypeError, "A.shape is not a tuple");
            return NULL;
        }
        if (A_nrows != p || A_ncols != n) {
            PyErr_SetString(PyExc_TypeError, "A must have compatible dimensions with b and c");
            return NULL;
        }
    }

    /* --- Get data pointers for c, h and (optionally) b --- */
    cpr = (qp_real *)PyArray_DATA(c);
    hpr = (qp_real *)PyArray_DATA(h);
    if (b)
        bpr = (qp_real *)PyArray_DATA(b);

    /* Get pointers for G's arrays */
    Gpr = (qp_real *)PyArray_DATA(G_data_arr);
    Gir = (qp_int *)PyArray_DATA(G_indices_arr);
    Gjc = (qp_int *)PyArray_DATA(G_indptr_arr);

    if (A && b) {
        Apr = (qp_real *)PyArray_DATA(A_data_arr);
        Air = (qp_int *)PyArray_DATA(A_indices_arr);
        Ajc = (qp_int *)PyArray_DATA(A_indptr_arr);
    }

    /* --- Process options --- */
    settings inopts;
    inopts.abstol = ABSTOL;
    inopts.reltol = RELTOL;
    inopts.maxit = MAXIT;
    inopts.sigma = SIGMA;
    inopts.verbose = VERBOSE;

    if (opts && PyDict_Check(opts)) {
        opts_maxiter = PyDict_GetItemString(opts, "MAXITER");
        if (opts_maxiter) {
            Py_INCREF(opts_maxiter);
            if (qpLong_check(opts_maxiter)) {
                inopts.maxit = (qp_int)qp_getlong(opts_maxiter);
            }
            Py_DECREF(opts_maxiter);
        }
        opts_abstol = PyDict_GetItemString(opts, "ABSTOL");
        if (opts_abstol) {
            Py_INCREF(opts_abstol);
            if (PyFloat_Check(opts_abstol)) {
                inopts.abstol = (qp_real)PyFloat_AsDouble(opts_abstol);
            }
            Py_DECREF(opts_abstol);
        }
        opts_reltol = PyDict_GetItemString(opts, "RELTOL");
        if (opts_reltol) {
            Py_INCREF(opts_reltol);
            if (PyFloat_Check(opts_reltol)) {
                inopts.reltol = (qp_real)PyFloat_AsDouble(opts_reltol);
            }
            Py_DECREF(opts_reltol);
        }
        opts_sigma = PyDict_GetItemString(opts, "SIGMA");
        if (opts_sigma) {
            Py_INCREF(opts_sigma);
            if (PyFloat_Check(opts_sigma)) {
                inopts.sigma = (qp_real)PyFloat_AsDouble(opts_sigma);
            }
            Py_DECREF(opts_sigma);
        }
        opts_verbose = PyDict_GetItemString(opts, "VERBOSE");
        if (opts_verbose) {
            Py_INCREF(opts_verbose);
            if (qpLong_check(opts_verbose)) {
                inopts.verbose = (qp_int)qp_getlong(opts_verbose);
            }
            Py_DECREF(opts_verbose);
        }
        opts_output = PyDict_GetItemString(opts, "OUTPUT");
        if (opts_output) {
            Py_INCREF(opts_output);
            if (qpLong_check(opts_output)) {
                opts_output_level = (qp_int)qp_getlong(opts_output);
            }
            Py_DECREF(opts_output);
        }
    }

    /* --- Call QP_SETUP using constructed CSC arrays for P --- */
    QP *myQP;
    myQP = QP_SETUP(n, m, p,
                    (qp_int *)PyArray_DATA(P_indptr_arr_new), // Pjc from diagonal CSC
                    (qp_int *)PyArray_DATA(P_indices_arr_new),  // Pir from diagonal CSC
                    Ppr,                                        // Diagonal values from P
                    Ajc, Air, Apr,                              // A’s arrays (may be NULL if A not provided)
                    (qp_int *)PyArray_DATA(G_indptr_arr),       // Gjc
                    (qp_int *)PyArray_DATA(G_indices_arr),        // Gir
                    Gpr,
                    cpr, hpr, bpr, 0.0, NULL);

    /* Copy options */
    myQP->options->abstol = inopts.abstol;
    myQP->options->reltol = inopts.reltol;
    myQP->options->maxit = inopts.maxit;
    myQP->options->sigma = inopts.sigma;
    myQP->options->verbose = inopts.verbose;

    /* --- Solve the QP --- */
    qp_int ExitCode;
    Py_BEGIN_ALLOW_THREADS;
    ExitCode = QP_SOLVE(myQP);
    Py_END_ALLOW_THREADS;

    /* --- Prepare results --- */
    sol_xdim[0] = myQP->n;
    sol_x = (PyArrayObject *)PyArray_SimpleNew(1, sol_xdim, NPY_DOUBLE);
    memcpy(PyArray_DATA(sol_x), myQP->x, myQP->n * sizeof(qp_real));

    switch (opts_output_level) {
        case 1:
            basic_info = Py_BuildValue("{s:l,s:l,s:d,s:d}",
                                       "ExitFlag", ExitCode,
                                       "Iterations", myQP->stats->IterationCount,
                                       "Setup_Time", myQP->stats->tsetup,
                                       "Solve_Time", myQP->stats->tsolve);
            result = Py_BuildValue("{s:O,s:O}",
                                   "sol", sol_x,
                                   "basicInfo", basic_info);
            Py_DECREF(basic_info);
            break;
        case 2:
            basic_info = Py_BuildValue("{s:l,s:l,s:d,s:d}",
                                       "ExitFlag", ExitCode,
                                       "Iterations", myQP->stats->IterationCount,
                                       "Setup_Time", myQP->stats->tsetup,
                                       "Solve_Time", myQP->stats->tsolve);
            if (b && A) {
                sol_ydim[0] = myQP->p;
                sol_y = (PyArrayObject *)PyArray_SimpleNew(1, sol_ydim, NPY_DOUBLE);
                memcpy(PyArray_DATA(sol_y), myQP->y, myQP->p * sizeof(qp_real));
            }
            sol_zdim[0] = myQP->m;
            sol_z = (PyArrayObject *)PyArray_SimpleNew(1, sol_zdim, NPY_DOUBLE);
            memcpy(PyArray_DATA(sol_z), myQP->z, myQP->m * sizeof(qp_real));
            sol_sdim[0] = myQP->m;
            sol_s = (PyArrayObject *)PyArray_SimpleNew(1, sol_sdim, NPY_DOUBLE);
            memcpy(PyArray_DATA(sol_s), myQP->s, myQP->m * sizeof(qp_real));
            if (b && A) {
                adv_info = Py_BuildValue("{s:d,s:d,s:d,s:O,s:O,s:O}",
                                         "fval", myQP->stats->fval,
                                         "kktTime", myQP->stats->kkt_time,
                                         "ldlTime", myQP->stats->ldl_numeric,
                                         "y", sol_y,
                                         "z", sol_z,
                                         "s", sol_s);
            } else {
                adv_info = Py_BuildValue("{s:d,s:d,s:d,s:O,s:O}",
                                         "fval", myQP->stats->fval,
                                         "kktTime", myQP->stats->kkt_time,
                                         "ldlTime", myQP->stats->ldl_numeric,
                                         "z", sol_z,
                                         "s", sol_s);
            }
            result = Py_BuildValue("{s:O,s:O,s:O}",
                                   "sol", sol_x,
                                   "basicInfo", basic_info,
                                   "advInfo", adv_info);
            Py_DECREF(basic_info);
            Py_DECREF(adv_info);
            if (sol_y) {
                Py_DECREF(sol_y);
            }
            Py_DECREF(sol_z);
            Py_DECREF(sol_s);
            break;
        default:
            result = Py_BuildValue("{s:O}",
                                   "sol", sol_x);
            break;
    }
    Py_DECREF(sol_x);

    /* --- Clean up --- */
    Py_DECREF(P_data_arr);
    Py_DECREF(P_indices_arr_new);
    Py_DECREF(P_indptr_arr_new);
    Py_DECREF(G_data_arr);
    Py_DECREF(G_indices_arr);
    Py_DECREF(G_indptr_arr);
    if (A_data_arr) {
        Py_DECREF(A_data_arr);
        Py_DECREF(A_indices_arr);
        Py_DECREF(A_indptr_arr);
    }

    QP_CLEANUP(myQP);
    return result;
}

static PyObject *method_qp_SWIFT_sparse_with_box(PyObject *self, PyObject *args, PyObject *kwargs)
{
    /* Input arrays and objects */
    PyArrayObject *c, *h, *h_box, *b = NULL;
    PyObject *P, *G, *G_box, *A = NULL;
    PyObject *opts = NULL;

    /* For constructing the CSC representation for P */
    PyArrayObject *P_data_arr, *P_indptr_arr_new, *P_indices_arr_new;

    /* For G_box extraction (expects a scipy.sparse.csc_matrix) */
    PyObject *G_box_data_obj, *G_box_indices_obj, *G_box_indptr_obj, *G_box_shape_obj;
    PyArrayObject *G_box_data_arr, *G_box_indices_arr, *G_box_indptr_arr;

    /* For G extraction (expects a scipy.sparse.csc_matrix) */
    PyObject *G_data_obj, *G_indices_obj, *G_indptr_obj, *G_shape_obj;
    PyArrayObject *G_data_arr, *G_indices_arr, *G_indptr_arr;

    /* Combined (stacked) G arrays and combined h vector */
    PyArrayObject *G_combined_data_arr, *G_combined_indices_arr, *G_combined_indptr_arr;
    PyArrayObject *h_combined_arr;

    /* For A extraction if provided */
    PyObject *A_data_obj = NULL, *A_indices_obj = NULL, *A_indptr_obj = NULL, *A_shape_obj = NULL;
    PyArrayObject *A_data_arr = NULL, *A_indices_arr = NULL, *A_indptr_arr = NULL;

    /* Pointers to data */
    qp_real *cpr, *hpr, *bpr = NULL;
    qp_real *Ppr, *Gpr, *Apr = NULL;
    qp_int *Pjc, *Pir;
    qp_int *Gjc, *Gir;
    qp_int *Ajc = NULL, *Air = NULL;

    PyObject *basic_info = NULL, *adv_info = NULL, *result;

    /* Results arrays */
    PyArrayObject *sol_x, *sol_y = NULL, *sol_z = NULL, *sol_s = NULL;
    npy_intp sol_xdim[1], sol_ydim[1], sol_zdim[1], sol_sdim[1];

    /* Options */
    PyObject *opts_maxiter = NULL, *opts_abstol = NULL, *opts_reltol = NULL;
    PyObject *opts_sigma = NULL, *opts_verbose = NULL, *opts_output = NULL;
    qp_int opts_output_level = 10;

    /* Dimensions and temporary variables */
    qp_int n, m, p = 0;
    qp_int m_box, m_bA;
    qp_int G_nrows, G_ncols;
    qp_int G_box_nrows, G_box_ncols;
    qp_int nnz_G, nnz_G_box, nnz_total;
    qp_int col_nnz;
    qp_int A_nrows = 0, A_ncols = 0;

    static char *kwlist[] = {"c", "h_box", "h", "P", "G_box", "G", "A", "b", "opts", NULL};

    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!OOO|OOO", kwlist,
                                     &PyArray_Type, &c,
                                     &PyArray_Type, &h_box,
                                     &PyArray_Type, &h,
                                     &P,
                                     &G_box,
                                     &G,
                                     &A,
                                     &b,
                                     &opts))
    {
        return NULL;
    }

    /* Validate c */
    if (!PyArray_ISFLOAT(c) || (PyArray_NDIM(c) != 1)) {
        PyErr_SetString(PyExc_TypeError, "c must be a floating array with one dimension");
        return NULL;
    }
    n = (qp_int)PyArray_DIM(c, 0);

    /* For inequality constraints, we now expect the box constraints first */
    if (!PyArray_ISFLOAT(h_box) || (PyArray_NDIM(h_box) != 1)) {
        PyErr_SetString(PyExc_TypeError, "h_box must be a floating array with one dimension");
        return NULL;
    }
    if (!PyArray_ISFLOAT(h) || (PyArray_NDIM(h) != 1)) {
        PyErr_SetString(PyExc_TypeError, "h must be a floating array with one dimension");
        return NULL;
    }
    /* m_box = number of box constraints, m_bA = number of “regular” inequalities */
    m_box = (qp_int)PyArray_DIM(h_box, 0);
    m_bA = (qp_int)PyArray_DIM(h, 0);
    m = m_box + m_bA;  /* total number of inequality constraints */

    /* Check b and A */
    if (b && A) {
        if (!PyArray_ISFLOAT(b) || (PyArray_NDIM(b) != 1)) {
            PyErr_SetString(PyExc_TypeError, "b must be a floating array with one dimension");
            return NULL;
        }
        p = (qp_int)PyArray_DIM(b, 0);
    }

    /* --- Process P as a 1D array holding the diagonal entries --- */
    if (!PyArray_Check(P) || !PyArray_ISFLOAT(P) || (PyArray_NDIM(P) != 1)) {
         PyErr_SetString(PyExc_TypeError, "P must be a floating array with one dimension");
         return NULL;
    }
    if (PyArray_DIM(P, 0) != n) {
         PyErr_SetString(PyExc_ValueError, "Length of P must equal length of c");
         return NULL;
    }
    P_data_arr = (PyArrayObject *)PyArray_FROMANY(P, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    if (!P_data_arr) {
         PyErr_SetString(PyExc_TypeError, "Error converting P to a numpy array");
         return NULL;
    }
    Ppr = (qp_real *)PyArray_DATA(P_data_arr);

    /* Build the CSC representation for the diagonal matrix:
       - P_indptr: [0, 1, 2, ..., n]
       - P_indices: [0, 1, 2, ..., n-1] */
    {
         npy_intp dims_indptr[1] = {n + 1};
         P_indptr_arr_new = (PyArrayObject *)PyArray_SimpleNew(1, dims_indptr, NPY_INT64);
         if (!P_indptr_arr_new) {
              Py_DECREF(P_data_arr);
              PyErr_SetString(PyExc_MemoryError, "Could not allocate P_indptr array");
              return NULL;
         }
         Pjc = (qp_int *)PyArray_DATA(P_indptr_arr_new);
         for (qp_int j = 0; j <= n; j++) {
              Pjc[j] = j;
         }
    }
    {
         npy_intp dims_indices[1] = {n};
         P_indices_arr_new = (PyArrayObject *)PyArray_SimpleNew(1, dims_indices, NPY_INT64);
         if (!P_indices_arr_new) {
              Py_DECREF(P_data_arr);
              Py_DECREF(P_indptr_arr_new);
              PyErr_SetString(PyExc_MemoryError, "Could not allocate P_indices array");
              return NULL;
         }
         Pir = (qp_int *)PyArray_DATA(P_indices_arr_new);
         for (qp_int j = 0; j < n; j++) {
              Pir[j] = j;
         }
    }
    /* --- End P processing --- */

    /* --- Process G as a CSC matrix --- */
    G_data_obj = PyObject_GetAttrString(G, "data");
    G_indices_obj = PyObject_GetAttrString(G, "indices");
    G_indptr_obj = PyObject_GetAttrString(G, "indptr");
    G_shape_obj = PyObject_GetAttrString(G, "shape");

    if (!G_data_obj || !G_indices_obj || !G_indptr_obj || !G_shape_obj) {
        PyErr_SetString(PyExc_TypeError, "G must be a scipy.sparse.csc_matrix with data, indices, indptr, shape attributes");
        return NULL;
    }
    G_data_arr = (PyArrayObject *)PyArray_FROMANY(G_data_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    G_indices_arr = (PyArrayObject *)PyArray_FROMANY(G_indices_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    G_indptr_arr = (PyArrayObject *)PyArray_FROMANY(G_indptr_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);

    if (PyTuple_Check(G_shape_obj)) {
        G_nrows = (qp_int)PyLong_AsLong(PyTuple_GetItem(G_shape_obj, 0));
        G_ncols = (qp_int)PyLong_AsLong(PyTuple_GetItem(G_shape_obj, 1));
    } else {
        PyErr_SetString(PyExc_TypeError, "G.shape is not a tuple");
        return NULL;
    }
    /* Now, G is assumed to correspond to the regular inequalities (second block) */
    if (G_nrows != m_bA || G_ncols != n) {
        PyErr_SetString(PyExc_TypeError, "G must have dimensions of h and c");
        return NULL;
    }

    /* --- Process G_box as a CSC matrix --- */
    G_box_data_obj = PyObject_GetAttrString(G_box, "data");
    G_box_indices_obj = PyObject_GetAttrString(G_box, "indices");
    G_box_indptr_obj = PyObject_GetAttrString(G_box, "indptr");
    G_box_shape_obj = PyObject_GetAttrString(G_box, "shape");

    if (!G_box_data_obj || !G_box_indices_obj || !G_box_indptr_obj || !G_box_shape_obj) {
        PyErr_SetString(PyExc_TypeError, "G_box must be a scipy.sparse.csc_matrix with data, indices, indptr, shape attributes");
        return NULL;
    }
    G_box_data_arr = (PyArrayObject *)PyArray_FROMANY(G_box_data_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
    G_box_indices_arr = (PyArrayObject *)PyArray_FROMANY(G_box_indices_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
    G_box_indptr_arr = (PyArrayObject *)PyArray_FROMANY(G_box_indptr_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);

    if (PyTuple_Check(G_box_shape_obj)) {
        G_box_nrows = (qp_int)PyLong_AsLong(PyTuple_GetItem(G_box_shape_obj, 0));
        G_box_ncols = (qp_int)PyLong_AsLong(PyTuple_GetItem(G_box_shape_obj, 1));
    } else {
        PyErr_SetString(PyExc_TypeError, "G_box.shape is not a tuple");
        return NULL;
    }
    /* G_box is assumed to correspond to the box constraints (first block) */
    if (G_box_nrows != m_box || G_box_ncols != n) {
        PyErr_SetString(PyExc_TypeError, "G_box must have dimensions of h_box and c");
        return NULL;
    }

    /* --- Build combined G = [G_box; G] in CSC format --- */
    nnz_G = (qp_int)PyArray_DIM(G_data_arr, 0);
    nnz_G_box = (qp_int)PyArray_DIM(G_box_data_arr, 0);
    nnz_total = nnz_G + nnz_G_box;
    {
        npy_intp dims_nnz[1] = {nnz_total};
        G_combined_data_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims_nnz, NPY_DOUBLE);
        G_combined_indices_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims_nnz, NPY_INT64);
    }
    {
        npy_intp dims_indptr[1] = {n + 1};
        G_combined_indptr_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims_indptr, NPY_INT64);
    }
    /* Initialize first entry of indptr */
    ((qp_int *)PyArray_DATA(G_combined_indptr_arr))[0] = 0;
    for (qp_int j = 0; j < n; j++) {
         qp_int start_G_box = ((qp_int *)PyArray_DATA(G_box_indptr_arr))[j];
         qp_int end_G_box = ((qp_int *)PyArray_DATA(G_box_indptr_arr))[j+1];
         qp_int start_G = ((qp_int *)PyArray_DATA(G_indptr_arr))[j];
         qp_int end_G = ((qp_int *)PyArray_DATA(G_indptr_arr))[j+1];
         col_nnz = (end_G_box - start_G_box) + (end_G - start_G);
         ((qp_int *)PyArray_DATA(G_combined_indptr_arr))[j+1] =
                ((qp_int *)PyArray_DATA(G_combined_indptr_arr))[j] + col_nnz;

         /* First, copy data and indices from G_box (no offset) */
         for (qp_int k = start_G_box; k < end_G_box; k++) {
              qp_int pos = ((qp_int *)PyArray_DATA(G_combined_indptr_arr))[j] + (k - start_G_box);
              ((qp_real *)PyArray_DATA(G_combined_data_arr))[pos] = ((qp_real *)PyArray_DATA(G_box_data_arr))[k];
              ((qp_int *)PyArray_DATA(G_combined_indices_arr))[pos] = ((qp_int *)PyArray_DATA(G_box_indices_arr))[k];
         }
         /* Then, copy data and indices from G, offsetting row indices by m_box */
         for (qp_int k = start_G; k < end_G; k++) {
              qp_int pos = ((qp_int *)PyArray_DATA(G_combined_indptr_arr))[j] + (end_G_box - start_G_box) + (k - start_G);
              ((qp_real *)PyArray_DATA(G_combined_data_arr))[pos] = ((qp_real *)PyArray_DATA(G_data_arr))[k];
              ((qp_int *)PyArray_DATA(G_combined_indices_arr))[pos] = ((qp_int *)PyArray_DATA(G_indices_arr))[k] + m_box;
         }
    }

    /* --- Build combined h = [h_box; h] --- */
    {
         npy_intp dims_h_total[1] = {m};
         h_combined_arr = (PyArrayObject *)PyArray_SimpleNew(1, dims_h_total, NPY_DOUBLE);
         memcpy(PyArray_DATA(h_combined_arr), PyArray_DATA(h_box), m_box * sizeof(qp_real));
         memcpy((qp_real *)PyArray_DATA(h_combined_arr) + m_box, PyArray_DATA(h), m_bA * sizeof(qp_real));
    }

    /* --- Process A if provided --- */
    if (A && b) {
        A_data_obj = PyObject_GetAttrString(A, "data");
        A_indices_obj = PyObject_GetAttrString(A, "indices");
        A_indptr_obj = PyObject_GetAttrString(A, "indptr");
        A_shape_obj = PyObject_GetAttrString(A, "shape");

        if (!A_data_obj || !A_indices_obj || !A_indptr_obj || !A_shape_obj) {
            PyErr_SetString(PyExc_TypeError, "A must be a scipy.sparse.csc_matrix with data, indices, indptr, shape attributes");
            return NULL;
        }
        A_data_arr = (PyArrayObject *)PyArray_FROMANY(A_data_obj, NPY_DOUBLE, 1, 1, NPY_ARRAY_IN_ARRAY);
        A_indices_arr = (PyArrayObject *)PyArray_FROMANY(A_indices_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);
        A_indptr_arr = (PyArrayObject *)PyArray_FROMANY(A_indptr_obj, NPY_INT64, 1, 1, NPY_ARRAY_IN_ARRAY);

        if (PyTuple_Check(A_shape_obj)) {
            A_nrows = (qp_int)PyLong_AsLong(PyTuple_GetItem(A_shape_obj, 0));
            A_ncols = (qp_int)PyLong_AsLong(PyTuple_GetItem(A_shape_obj, 1));
        } else {
            PyErr_SetString(PyExc_TypeError, "A.shape is not a tuple");
            return NULL;
        }
        if (A_nrows != p || A_ncols != n) {
            PyErr_SetString(PyExc_TypeError, "A must have compatible dimensions with b and c");
            return NULL;
        }
    }

    /* --- Get data pointers for c, and (optionally) b --- */
    cpr = (qp_real *)PyArray_DATA(c);
    hpr = (qp_real *)PyArray_DATA(h_combined_arr);
    if (b)
        bpr = (qp_real *)PyArray_DATA(b);

    /* Get pointers for G's arrays */
    Gpr = (qp_real *)PyArray_DATA(G_combined_data_arr);
    Gir = (qp_int *)PyArray_DATA(G_combined_indices_arr);
    Gjc = (qp_int *)PyArray_DATA(G_combined_indptr_arr);

    if (A && b) {
        Apr = (qp_real *)PyArray_DATA(A_data_arr);
        Air = (qp_int *)PyArray_DATA(A_indices_arr);
        Ajc = (qp_int *)PyArray_DATA(A_indptr_arr);
    }

    /* --- Process options --- */
    settings inopts;
    inopts.abstol = ABSTOL;
    inopts.reltol = RELTOL;
    inopts.maxit = MAXIT;
    inopts.sigma = SIGMA;
    inopts.verbose = VERBOSE;

    if (opts && PyDict_Check(opts)) {
        opts_maxiter = PyDict_GetItemString(opts, "MAXITER");
        if (opts_maxiter) {
            Py_INCREF(opts_maxiter);
            if (qpLong_check(opts_maxiter)) {
                inopts.maxit = (qp_int)qp_getlong(opts_maxiter);
            }
            Py_DECREF(opts_maxiter);
        }
        opts_abstol = PyDict_GetItemString(opts, "ABSTOL");
        if (opts_abstol) {
            Py_INCREF(opts_abstol);
            if (PyFloat_Check(opts_abstol)) {
                inopts.abstol = (qp_real)PyFloat_AsDouble(opts_abstol);
            }
            Py_DECREF(opts_abstol);
        }
        opts_reltol = PyDict_GetItemString(opts, "RELTOL");
        if (opts_reltol) {
            Py_INCREF(opts_reltol);
            if (PyFloat_Check(opts_reltol)) {
                inopts.reltol = (qp_real)PyFloat_AsDouble(opts_reltol);
            }
            Py_DECREF(opts_reltol);
        }
        opts_sigma = PyDict_GetItemString(opts, "SIGMA");
        if (opts_sigma) {
            Py_INCREF(opts_sigma);
            if (PyFloat_Check(opts_sigma)) {
                inopts.sigma = (qp_real)PyFloat_AsDouble(opts_sigma);
            }
            Py_DECREF(opts_sigma);
        }
        opts_verbose = PyDict_GetItemString(opts, "VERBOSE");
        if (opts_verbose) {
            Py_INCREF(opts_verbose);
            if (qpLong_check(opts_verbose)) {
                inopts.verbose = (qp_int)qp_getlong(opts_verbose);
            }
            Py_DECREF(opts_verbose);
        }
        opts_output = PyDict_GetItemString(opts, "OUTPUT");
        if (opts_output) {
            Py_INCREF(opts_output);
            if (qpLong_check(opts_output)) {
                opts_output_level = (qp_int)qp_getlong(opts_output);
            }
            Py_DECREF(opts_output);
        }
    }

    /* --- Call QP_SETUP using constructed CSC arrays for P, combined G, and combined h --- */
    QP *myQP;
    myQP = QP_SETUP(n, m, p,
                    (qp_int *)PyArray_DATA(P_indptr_arr_new),
                    (qp_int *)PyArray_DATA(P_indices_arr_new),
                    Ppr,
                    Ajc, Air, Apr,
                    (qp_int *)PyArray_DATA(G_combined_indptr_arr), (qp_int *)PyArray_DATA(G_combined_indices_arr), Gpr,
                    cpr, hpr, bpr, 0.0, NULL);

    /* Copy options */
    myQP->options->abstol = inopts.abstol;
    myQP->options->reltol = inopts.reltol;
    myQP->options->maxit = inopts.maxit;
    myQP->options->sigma = inopts.sigma;
    myQP->options->verbose = inopts.verbose;

    /* --- Solve the QP --- */
    qp_int ExitCode;
    Py_BEGIN_ALLOW_THREADS;
    ExitCode = QP_SOLVE(myQP);
    Py_END_ALLOW_THREADS;

    /* --- Prepare results --- */
    sol_xdim[0] = myQP->n;
    sol_x = (PyArrayObject *)PyArray_SimpleNew(1, sol_xdim, NPY_DOUBLE);
    memcpy(PyArray_DATA(sol_x), myQP->x, myQP->n * sizeof(qp_real));

    switch (opts_output_level) {
        case 1:
            basic_info = Py_BuildValue("{s:l,s:l,s:d,s:d}",
                                       "ExitFlag", ExitCode,
                                       "Iterations", myQP->stats->IterationCount,
                                       "Setup_Time", myQP->stats->tsetup,
                                       "Solve_Time", myQP->stats->tsolve);
            result = Py_BuildValue("{s:O,s:O}",
                                   "sol", sol_x,
                                   "basicInfo", basic_info);
            Py_DECREF(basic_info);
            break;
        case 2:
            basic_info = Py_BuildValue("{s:l,s:l,s:d,s:d}",
                                       "ExitFlag", ExitCode,
                                       "Iterations", myQP->stats->IterationCount,
                                       "Setup_Time", myQP->stats->tsetup,
                                       "Solve_Time", myQP->stats->tsolve);
            if (b && A) {
                sol_ydim[0] = myQP->p;
                sol_y = (PyArrayObject *)PyArray_SimpleNew(1, sol_ydim, NPY_DOUBLE);
                memcpy(PyArray_DATA(sol_y), myQP->y, myQP->p * sizeof(qp_real));
            }
            sol_zdim[0] = myQP->m;
            sol_z = (PyArrayObject *)PyArray_SimpleNew(1, sol_zdim, NPY_DOUBLE);
            memcpy(PyArray_DATA(sol_z), myQP->z, myQP->m * sizeof(qp_real));
            sol_sdim[0] = myQP->m;
            sol_s = (PyArrayObject *)PyArray_SimpleNew(1, sol_sdim, NPY_DOUBLE);
            memcpy(PyArray_DATA(sol_s), myQP->s, myQP->m * sizeof(qp_real));
            if (b && A) {
                adv_info = Py_BuildValue("{s:d,s:d,s:d,s:O,s:O,s:O}",
                                         "fval", myQP->stats->fval,
                                         "kktTime", myQP->stats->kkt_time,
                                         "ldlTime", myQP->stats->ldl_numeric,
                                         "y", sol_y,
                                         "z", sol_z,
                                         "s", sol_s);
            } else {
                adv_info = Py_BuildValue("{s:d,s:d,s:d,s:O,s:O}",
                                         "fval", myQP->stats->fval,
                                         "kktTime", myQP->stats->kkt_time,
                                         "ldlTime", myQP->stats->ldl_numeric,
                                         "z", sol_z,
                                         "s", sol_s);
            }
            result = Py_BuildValue("{s:O,s:O,s:O}",
                                   "sol", sol_x,
                                   "basicInfo", basic_info,
                                   "advInfo", adv_info);
            Py_DECREF(basic_info);
            Py_DECREF(adv_info);
            if (sol_y) {
                Py_DECREF(sol_y);
            }
            Py_DECREF(sol_z);
            Py_DECREF(sol_s);
            break;
        default:
            result = Py_BuildValue("{s:O}",
                                   "sol", sol_x);
            break;
    }
    Py_DECREF(sol_x);

    /* --- Clean up --- */
    Py_DECREF(P_data_arr);
    Py_DECREF(P_indices_arr_new);
    Py_DECREF(P_indptr_arr_new);
    Py_DECREF(G_data_arr);
    Py_DECREF(G_indices_arr);
    Py_DECREF(G_indptr_arr);
    Py_DECREF(G_box_data_arr);
    Py_DECREF(G_box_indices_arr);
    Py_DECREF(G_box_indptr_arr);
    Py_DECREF(G_box_data_obj);
    Py_DECREF(G_box_indices_obj);
    Py_DECREF(G_box_indptr_obj);
    Py_DECREF(G_box_shape_obj);
    Py_DECREF(G_data_obj);
    Py_DECREF(G_indices_obj);
    Py_DECREF(G_indptr_obj);
    Py_DECREF(G_shape_obj);
    if (A_data_arr) {
        Py_DECREF(A_data_arr);
        Py_DECREF(A_indices_arr);
        Py_DECREF(A_indptr_arr);
    }
    Py_DECREF(h_combined_arr);
    Py_DECREF(G_combined_data_arr);
    Py_DECREF(G_combined_indices_arr);
    Py_DECREF(G_combined_indptr_arr);

    QP_CLEANUP(myQP);
    return result;
}



static PyMethodDef qpSWIFTMethods[] = {
    {"run", method_qpSWIFT, METH_VARARGS | METH_KEYWORDS, "res = qpSWIFT.run(c,h,P,G,A,b,opts) \n"
                                                          "Please refer to the Python Documentation or qpSWIFT_help.py file \n"},
    {"run_sparse", method_qp_SWIFT_sparse, METH_VARARGS | METH_KEYWORDS, "res = qpSWIFT.run_sparse(c, h, P, G, A, b, opts) \n"
                                                                         "Run the solver with sparse matrix inputs (P, G, A) in CSC format.\n"},
    {"run_sparse_with_box_constraints", method_qp_SWIFT_sparse_with_box, METH_VARARGS | METH_KEYWORDS, "res = qpSWIFT.run_sparse(c, h, h_box, P, G, G_box, A, b, opts) \n"
                                                                         "Run the solver with sparse matrix inputs (P, G, G_box, A) in CSC format.\n"},
    {NULL, NULL, 0, NULL}
};



#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef qpSWIFTModule = {
    PyModuleDef_HEAD_INIT,
    "qpSWIFT",
    "A Sparse Quadratic Programming Solver",
    -1,
    qpSWIFTMethods};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_qpSWIFT(void)
{
    import_array();
    return PyModule_Create(&qpSWIFTModule);
}
#else
PyMODINIT_FUNC initqpSWIFT(void)
{
    import_array();
    Py_InitModule("qpSWIFT", qpSWIFTMethods);
}
#endif
