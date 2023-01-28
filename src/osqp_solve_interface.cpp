// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "osqp.h"
#include <Rcpp.h>
#include <memory>
#include <algorithm>
#include <vector>

using namespace Rcpp;

void extractMatrixData_new(arma::sp_mat &sm, std::vector<c_int> &iout, std::vector<c_int> &pout, std::vector<c_float> &xout);
void extractMatrixData(const Rcpp::S4& mat, std::vector<c_int>& iout, std::vector<c_int>& pout, std::vector<c_float>& xout);
void translateSettings(OSQPSettings* settings, const Rcpp::List& pars);
void mycleanup (OSQPWorkspace* x);
S4 toDgCMat(csc*);
arma::vec psi(arma::vec z, double k);
arma::sp_mat convert_to_diag_sp(arma::vec z);

//typedef Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> XPtrOsqpWork;

// predicate for testing if a value is below -OSQP_INFTY
bool below_osqp_neg_inf (c_float x) {
  return (x < -OSQP_INFTY);
}

// predicate for testing if a value is above OSQP_INFTY
bool above_osqp_inf (c_float x) {
  return (x > OSQP_INFTY);
}

SEXP osqpSetup_new(arma::sp_mat &P, arma::vec &q, arma::sp_mat &A, arma::vec &l, arma::vec &u)
{
  IntegerVector dimP(2);
  IntegerVector dimA(2);
  dimP[0] = P.n_rows;
  dimP[1] = P.n_cols;
  dimA[0] = A.n_rows;
  dimA[1] = A.n_cols;
  if (dimP[0] != dimP[1] || dimP[0] != dimA[1])
    stop("bug");

  std::vector<c_int> A_i, A_p, P_i, P_p;
  std::vector<c_float> A_x, P_x, qvec(q.size()), lvec(l.size()), uvec(u.size());

  extractMatrixData_new(P, P_i, P_p, P_x);
  extractMatrixData_new(A, A_i, A_p, A_x);

  std::copy(q.begin(), q.end(), qvec.begin());
  std::copy(l.begin(), l.end(), lvec.begin());
  std::copy(u.begin(), u.end(), uvec.begin());

  // Threshold lvec to range [-OSQP_INFTY, OSQP_INFTY]
  std::replace_if(lvec.begin(), lvec.end(), below_osqp_neg_inf, -OSQP_INFTY);
  std::replace_if(lvec.begin(), lvec.end(), above_osqp_inf, OSQP_INFTY);

  // Threshold uvec to range [-OSQP_INFTY, OSQP_INFTY]
  std::replace_if(uvec.begin(), uvec.end(), below_osqp_neg_inf, -OSQP_INFTY);
  std::replace_if(uvec.begin(), uvec.end(), above_osqp_inf, OSQP_INFTY);

  std::unique_ptr<OSQPSettings> settings(new OSQPSettings);
  osqp_set_ROC_settings(settings.get());

  std::unique_ptr<OSQPData> data(new OSQPData);

  data->n = static_cast<c_int>(dimP[0]);
  data->m = static_cast<c_int>(dimA[0]);
  data->P = csc_matrix(data->n, data->n, P_x.size(), P_x.data(), P_i.data(), P_p.data());
  data->q = qvec.data();
  data->A = csc_matrix(data->m, data->n, A_x.size(), A_x.data(), A_i.data(), A_p.data());
  data->l = lvec.data();
  data->u = uvec.data();

  OSQPWorkspace *workp;
  osqp_setup(&workp, data.get(), settings.get());

  Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> work(workp);

  return work;
}

SEXP osqpSetup(const S4& P, const NumericVector& q, const S4& A, const NumericVector& l, const NumericVector& u, const List& pars)
{
  IntegerVector dimP = P.slot("Dim");
  IntegerVector dimA = A.slot("Dim");
  int n = dimP[0];
  int m = dimA[0];
  if (n != dimP[1] || n != dimA[1]) stop("bug");
  std::vector<c_int> A_i, A_p, P_i, P_p;
  std::vector<c_float> A_x, P_x, qvec(q.size()), lvec(l.size()), uvec(u.size());

  extractMatrixData(P, P_i, P_p, P_x);
  extractMatrixData(A, A_i, A_p, A_x);

  std::copy(q.begin(), q.end(), qvec.begin());
  std::copy(l.begin(), l.end(), lvec.begin());
  std::copy(u.begin(), u.end(), uvec.begin());

  // Threshold lvec to range [-OSQP_INFTY, OSQP_INFTY]
  std::replace_if(lvec.begin(), lvec.end(), below_osqp_neg_inf, -OSQP_INFTY);
  std::replace_if(lvec.begin(), lvec.end(), above_osqp_inf, OSQP_INFTY);  

  // Threshold uvec to range [-OSQP_INFTY, OSQP_INFTY]  
  std::replace_if(uvec.begin(), uvec.end(), below_osqp_neg_inf, -OSQP_INFTY);
  std::replace_if(uvec.begin(), uvec.end(), above_osqp_inf, OSQP_INFTY);  

  std::unique_ptr<OSQPSettings> settings (new OSQPSettings);
  osqp_set_default_settings(settings.get());

  if (pars.size())
    translateSettings(settings.get(), pars);

  std::unique_ptr<OSQPData> data (new OSQPData);

  data->n = static_cast<c_int>(n);
  data->m = static_cast<c_int>(m);
  data->P = csc_matrix(data->n, data->n, P_x.size(), P_x.data(), P_i.data(), P_p.data());
  data->q = qvec.data();
  data->A = csc_matrix(data->m, data->n, A_x.size(), A_x.data(), A_i.data(), A_p.data());
  data->l = lvec.data();
  data->u = uvec.data();

  OSQPWorkspace* workp;
  osqp_setup(&workp, data.get(), settings.get());

    Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> work(workp);

  return work;
}


List osqpSolve(SEXP workPtr)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);
  c_int n = work->data->n;
  c_int m = work->data->m;
  c_int res = osqp_solve(work);



  std::string status = work->info->status;
  List info =  List::create(_("iter") = work->info->iter,
                      _("status") = status,
                      _("status_val") = work->info->status_val,
                      _("status_polish") = work->info->status_polish,
                      _("obj_val") = work->info->obj_val,
                      _("pri_res") = work->info->pri_res,
                      _("dua_res") = work->info->dua_res,
                      _("setup_time") = work->info->setup_time,
                      _("solve_time") = work->info->solve_time,
                      _("update_time") = work->info->update_time,
                      _("polish_time") = work->info->polish_time,
                      _("run_time") = work->info->run_time,
                      _("rho_estimate") = work->info->rho_estimate,
                      _("rho_updates") = work->info->rho_updates);

  List resl;
  if (res != OSQP_UNSOLVED)
  {
    NumericVector x(work->solution->x, work->solution->x + n);
    NumericVector y(work->solution->y, work->solution->y + m);
    NumericVector dx(work->delta_x, work->delta_x + n);
    NumericVector dy(work->delta_y, work->delta_y + m);
    resl = List::create(_("x") = x,
                       _("y") = y,
                       _("prim_inf_cert") = dx,
                       _("dual_inf_cert") = dy,
                       _("info") = info);
  }
  else
    resl = List::create(_("x") = NA_REAL,
                        _("y") = NA_REAL,
                        _("prim_inf_cert") = NA_REAL,
                        _("dual_inf_cert") = NA_REAL,
                        _("info") = info);

  return resl;
}

List osqpGetParams(SEXP workPtr)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);
  IntegerVector linsys;
  if (work->settings->linsys_solver == QDLDL_SOLVER)
    linsys = IntegerVector::create(_("QDLDL_SOLVER") = work->settings->linsys_solver);
  else if (work->settings->linsys_solver == MKL_PARDISO_SOLVER)
    linsys = IntegerVector::create(_("MKL_PARDISO_SOLVER") = work->settings->linsys_solver);
  else
    linsys = IntegerVector::create(_("UNKNOWN_SOLVER") = work->settings->linsys_solver);

  List res = List::create(_("rho") = work->settings->rho,
                          _("sigma") = work->settings->sigma,
                          _("max_iter") = work->settings->max_iter,
                          _("eps_abs") = work->settings->eps_abs,
                          _("eps_rel") = work->settings->eps_rel,
                          _("eps_prim_inf") = work->settings->eps_prim_inf,
                          _("eps_dual_inf") = work->settings->eps_dual_inf,
                          _("alpha") = work->settings->alpha,
                          _("linsys_solver") = linsys,
                          _("delta") = work->settings->delta,
                          _("polish") = (bool)work->settings->polish,
                          _("polish_refine_iter") = work->settings->polish_refine_iter,
                          _("verbose") = (bool)work->settings->verbose,
                          _("scaled_termination") = (bool)work->settings->scaled_termination,
                          _("check_termination") = work->settings->check_termination,
                          _("warm_start") = (bool)work->settings->warm_start,
                          _("scaling") = work->settings->scaling,
                          _("adaptive_rho") = work->settings->adaptive_rho,
                          _("adaptive_rho_interval") = work->settings->adaptive_rho_interval,
                          _("adaptive_rho_tolerance") = work->settings->adaptive_rho_tolerance);

  res.push_back(work->settings->adaptive_rho_fraction, "adaptive_rho_fraction");

  return res;
}




IntegerVector osqpGetDims(SEXP workPtr)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);
  auto res = IntegerVector::create(_("n") = work->data->n,
                                   _("m") = work->data->m);


  return res;
}

void osqpUpdate(SEXP workPtr,
    Rcpp::Nullable<NumericVector> q_new,
    Rcpp::Nullable<NumericVector> l_new,
    Rcpp::Nullable<NumericVector> u_new,
    Rcpp::Nullable<NumericVector> Px,
    Rcpp::Nullable<IntegerVector> Px_idx,
    Rcpp::Nullable<NumericVector> Ax,
    Rcpp::Nullable<IntegerVector> Ax_idx)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);

  // Update problem vectors
  if (q_new.isNotNull()) {
    osqp_update_lin_cost(work, as<NumericVector>(q_new.get()).begin());
  }
  if (l_new.isNotNull() && u_new.isNull()) {
    osqp_update_lower_bound(work, as<NumericVector>(l_new.get()).begin());
  }
  if (u_new.isNotNull() && l_new.isNull()) {
    osqp_update_upper_bound(work, as<NumericVector>(u_new.get()).begin());
  }
  if (u_new.isNotNull() && l_new.isNotNull()) {
    osqp_update_bounds(work,
        as<NumericVector>(l_new.get()).begin(),
        as<NumericVector>(u_new.get()).begin());
  }


  // Update problem matrices
  c_int * Px_idx_ = OSQP_NULL;
  c_int len_Px = 0;
  c_int * Ax_idx_ = OSQP_NULL;
  c_int len_Ax = 0;
  // Get which parameters are null
  if (Px_idx.isNotNull()) {
    Px_idx_ = (c_int *)as<IntegerVector>(Px_idx.get()).begin();
    NumericVector Px_ = Px.get();
    len_Px = Px_.size();
  }
  if (Ax_idx.isNotNull()) {
    Ax_idx_ = (c_int *)as<IntegerVector>(Ax_idx.get()).begin();
    NumericVector Ax_ = Ax.get();
    len_Ax = Ax_.size();
  }
  // Only P
  if (Px.isNotNull() && Ax.isNull()){
      osqp_update_P(work,
          as<NumericVector>(Px.get()).begin(),
          Px_idx_,
          len_Px);
  }

  // Only A
  if (Ax.isNotNull() && Px.isNull()){
      osqp_update_A(work, as<NumericVector>(Ax.get()).begin(),
                    Ax_idx_,
                    len_Ax);
  }

  // Both A and P
  if (Px.isNotNull() && Ax.isNotNull()){
      osqp_update_P_A(
          work,
          as<NumericVector>(Px.get()).begin(),
          Px_idx_,
          len_Px,
          as<NumericVector>(Ax.get()).begin(),
          Ax_idx_,
          len_Ax);
  }


}


void extractMatrixData_new(arma::sp_mat &sm, std::vector<c_int> &iout, std::vector<c_int> &pout, std::vector<c_float> &xout)
{
  IntegerVector dim(2);
  dim[0] = sm.n_rows;
  dim[1] = sm.n_cols;

  arma::vec x(sm.n_nonzero); // create space for values, and copy
  arma::arrayops::copy(x.begin(), sm.values, sm.n_nonzero);
  std::vector<double> vx = arma::conv_to<std::vector<double>>::from(x);

  arma::urowvec i(sm.n_nonzero); // create space for row_indices, and copy & cast
  arma::arrayops::copy(i.begin(), sm.row_indices, sm.n_nonzero);
  std::vector<int> vi = arma::conv_to<std::vector<int>>::from(i);

  arma::urowvec p(sm.n_cols + 1); // create space for col_ptrs, and copy
  arma::arrayops::copy(p.begin(), sm.col_ptrs, sm.n_cols + 1);
  // do not copy sentinel for returning R
  std::vector<int> vp = arma::conv_to<std::vector<int>>::from(p);

  iout.resize(vi.size());
  pout.resize(vp.size());
  xout.resize(vx.size());
  std::copy(vi.begin(), vi.end(), iout.begin());
  std::copy(vp.begin(), vp.end(), pout.begin());
  std::copy(vx.begin(), vx.end(), xout.begin());

  return;
}

void extractMatrixData(const S4& mat, std::vector<c_int>& iout, std::vector<c_int>& pout, std::vector<c_float>& xout)
{
  IntegerVector i = mat.slot("i");
  IntegerVector p = mat.slot("p");
  NumericVector x = mat.slot("x");

  iout.resize(i.size());
  pout.resize(p.size());
  xout.resize(x.size());
  std::copy(i.begin(), i.end(), iout.begin());
  std::copy(p.begin(), p.end(), pout.begin());
  std::copy(x.begin(), x.end(), xout.begin());

  return;
}


void translateSettings(OSQPSettings* settings, const List& pars)
{

  CharacterVector nms(pars.names());
  for (int i = 0; i < pars.size(); i++)
  {
    if (Rf_isNull(nms[i]))
      continue;
    auto nm = as<std::string>(nms[i]);

    if (nm == "rho")
      settings->rho = as<c_float>(pars[i]);
    else if (nm == "sigma")
      settings->sigma = as<c_float>(pars[i]);
    else if (nm == "eps_abs")
      settings->eps_abs = as<c_float>(pars[i]);
    else if (nm == "eps_rel")
      settings->eps_rel = as<c_float>(pars[i]);
    else if (nm == "eps_prim_inf")
      settings->eps_prim_inf = as<c_float>(pars[i]);
    else if (nm == "eps_dual_inf")
      settings->eps_dual_inf = as<c_float>(pars[i]);
    else if (nm == "alpha")
      settings->alpha = as<c_float>(pars[i]);
    else if (nm == "delta")
      settings->delta = as<c_float>(pars[i]);
    else if (nm == "adaptive_rho_fraction")
      settings->adaptive_rho_fraction = as<c_float>(pars[i]);
    else if (nm == "adaptive_rho_tolerance")
      settings->adaptive_rho_tolerance = as<c_float>(pars[i]);


    else if (nm == "linsys_solver")
      settings->linsys_solver = (linsys_solver_type)as<c_int>(pars[i]);
    else if (nm == "polish_refine_iter")
      settings->polish_refine_iter = as<c_int>(pars[i]);
    else if (nm == "check_termination")
      settings->check_termination = as<c_int>(pars[i]);
    else if (nm == "scaling")
      settings->scaling = as<c_int>(pars[i]);
    else if (nm == "max_iter")
      settings->max_iter = as<c_int>(pars[i]);
    else if (nm == "adaptive_rho")
      settings->adaptive_rho = as<c_int>(pars[i]);
    else if (nm == "adaptive_rho_interval")
      settings->adaptive_rho_interval = as<c_int>(pars[i]);


    else if (nm == "polish")
      settings->polish = as<c_int>(pars[i]);
    else if (nm == "verbose")
      settings->verbose = as<c_int>(pars[i]);
    else if (nm == "scaled_termination")
      settings->scaled_termination = as<c_int>(pars[i]);
    else if (nm == "warm_start")
      settings->warm_start = as<c_int>(pars[i]);
  }

  return;
}


void osqpUpdateSettings(SEXP workPtr, SEXP val, std::string nm)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);

  if (nm == "check_termination")
    osqp_update_check_termination(work, as<c_int>(val));
  else if (nm == "max_iter")
    osqp_update_max_iter(work, as<c_int>(val));
  else if (nm == "polish")
    osqp_update_polish(work, as<c_int>(val));
  else if (nm == "polish_refine_iter")
    osqp_update_polish_refine_iter(work, as<c_int>(val));
  else if (nm == "rho")
    osqp_update_rho(work, as<c_float>(val));
  else if (nm == "scaled_termination")
    osqp_update_scaled_termination(work, as<c_int>(val));
  else if (nm == "verbose")
    osqp_update_verbose(work, as<c_int>(val));
  else if (nm == "warm_start")
    osqp_update_warm_start(work, as<c_int>(val));
  else if (nm == "alpha")
    osqp_update_alpha(work, as<c_float>(val));
  else if (nm == "delta")
    osqp_update_delta(work, as<c_float>(val));
  else if (nm == "eps_abs")
    osqp_update_eps_abs(work, as<c_float>(val));
  else if (nm == "eps_dual_inf")
    osqp_update_eps_dual_inf(work, as<c_float>(val));
  else if (nm == "eps_prim_inf")
    osqp_update_eps_prim_inf(work, as<c_float>(val));
  else if (nm == "eps_rel")
    osqp_update_eps_rel(work, as<c_float>(val));
  else
    Rcout << "Param " + nm + " cannot be updated live" << std::endl;

}


SEXP osqpGetData(SEXP workPtr, std::string nm)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);

  if (nm == "P")
    return toDgCMat(work->data->P);
  if (nm == "A")
    return toDgCMat(work->data->A);

  if (nm == "q")
  {
    int n = work->data->n;
    NumericVector q(work->data->q, work->data->q + n);
    return q;
  }
  if (nm == "l")
  {
    int n = work->data->m;
    NumericVector q(work->data->l, work->data->l + n);
    return q;
  }
  if (nm == "u")
  {
    int n = work->data->m;
    NumericVector q(work->data->u, work->data->u + n);
    return q;
  }


  return R_NilValue;
}

S4 toDgCMat(csc* inmat)
{
  S4 m("dgCMatrix");

  int nnz = inmat->nzmax;
  int nr = inmat->m;
  int nc = inmat->n;

  NumericVector x(inmat->x, inmat->x+nnz);
  IntegerVector i(inmat->i, inmat->i+nnz);
  IntegerVector p(inmat->p, inmat->p+nc+1);
  IntegerVector dim = IntegerVector::create(nr, nc);


  m.slot("i")   = i;
  m.slot("p")   = p;
  m.slot("x")   = x;
  m.slot("Dim") = dim;

  return m;
}


void mycleanup (OSQPWorkspace* x)
{
  osqp_cleanup(x);
}



void osqpWarmStart(SEXP workPtr, Rcpp::Nullable<NumericVector> x, Rcpp::Nullable<NumericVector> y)
{
  auto work = as<Rcpp::XPtr<OSQPWorkspace, Rcpp::PreserveStorage, mycleanup> >(workPtr);

  if(x.isNull() && y.isNull())
  {
    return;
  } else if (x.isNotNull() && y.isNotNull())
  {
    osqp_warm_start(work, as<NumericVector>(x.get()).begin(),as<NumericVector>(y.get()).begin());


  } else if (x.isNotNull())
  {
    osqp_warm_start_x(work, as<NumericVector>(x.get()).begin());
  } else {
    osqp_warm_start_y(work, as<NumericVector>(y.get()).begin());
  }
  return;
}


arma::vec psi(arma::vec z, double k)
{
  return (z > 1 / k) * 0 + (z < 0) * 1 + (z >= 0 && z <= 1 / k) % (1 - k * z);
}

arma::sp_mat convert_to_diag_sp(arma::vec z)
{
  arma::mat X = arma::diagmat(z);
  arma::sp_mat Y(X);
  return Y;
}

// [[Rcpp::export]]
List DCCP_ROC(arma::mat X, arma::vec Y, arma::vec beta_init, double b_init, double gamma, double psi_k, int max_iter_num, double max_rel_gap)
{
  int n = X.n_rows;
  int p = X.n_cols;
  arma::mat only_ones(n, 1, arma::fill::ones);
  arma::mat aug_X = arma::join_rows(X, only_ones);

  arma::uvec Y_pos_indicator_u = (Y > 0) * 1.0;
  arma::vec Y_pos_indicator = arma::conv_to<arma::vec>::from(Y_pos_indicator_u);

  arma::uvec Y_neg_indicator_u = (Y < 0) * 1.0;
  arma::vec Y_neg_indicator = arma::conv_to<arma::vec>::from(Y_neg_indicator_u);

  int n_pos = arma::sum(Y > 0);
  int n_neg = n - n_pos;

  arma::vec beta_before = beta_init;
  double b_before = b_init;

  arma::vec margin_before = Y % (X * beta_before + b_before);
  arma::vec psi_value_before = psi(margin_before, psi_k);
  arma::vec psi_value_before_pos = psi_value_before % Y_pos_indicator;
  arma::vec psi_value_before_neg = psi_value_before % Y_neg_indicator;
  double alpha = arma::sum(psi_value_before_neg) / n_neg;

  double fun_value_before = arma::sum(psi_value_before_pos) / n + gamma / 2 * arma::sum(arma::square(beta_before));

  arma::uvec indicator_u = (margin_before > 0) * 1.0;
  arma::vec indicator = arma::conv_to<arma::vec>::from(indicator_u);
  arma::uvec b_vec_u = (Y < 0) * 1.0;
  arma::vec b_vec = arma::conv_to<arma::vec>::from(b_vec_u);
  double c_value = arma::sum(b_vec % (1.0 - indicator));

  arma::mat X_sub = X.rows(arma::find(indicator > 0));
  arma::vec Y_sub = Y.elem(arma::find(indicator > 0));
  arma::mat aug_X_sub = aug_X.rows(arma::find(indicator > 0));
  int n_sub = X_sub.n_rows;

  // diagonal matrix P
  arma::vec P_diag(p + n_sub + 1, arma::fill::ones);
  P_diag.subvec(0, p - 1) *= n * gamma;
  P_diag.subvec(p, p + n_sub) *= 0;
  arma::sp_mat P = convert_to_diag_sp(P_diag);

  // linear vector q
  arma::uvec q_u(p + n_sub + 1, arma::fill::zeros);
  q_u.subvec(p + 1, p + n_sub) = (Y_sub > 0) * 1.0;
  arma::vec q = arma::conv_to<arma::vec>::from(q_u);

  // coef matrix A
  arma::mat A_lt = psi_k * (aug_X_sub.each_col() % Y_sub);
  arma::mat A_rt = arma::eye(n_sub, n_sub);
  arma::mat A_lm = arma::zeros(n_sub, p + 1);
  arma::mat A_rm = A_rt;
  arma::mat A_ll = arma::zeros(1, p + 1);
  arma::mat A_rl = -b_vec.elem(arma::find(indicator > 0));

  arma::mat A_dense = arma::join_cols(arma::join_rows(A_lt, A_rt), arma::join_rows(A_lm, A_rm), arma::join_rows(A_ll, A_rl));
  arma::sp_mat A(A_dense);

  // lower bound vector l
  arma::vec l(2 * n_sub + 1, arma::fill::ones);
  l.subvec(n_sub, 2 * n_sub - 1) *= 0;
  l(2 * n_sub) = c_value - alpha * n_neg;

  // upper bound vector u
  arma::vec u(2 * n_sub + 1, arma::fill::ones);
  u *= OSQP_INFTY;

  SEXP problem = osqpSetup_new(P, q, A, l, u);
  List res = osqpSolve(problem);
  if (NumericVector::is_na(res[0]))
    stop("bug");

  arma::vec osqp_x = as<NumericVector>(res[0]);
  arma::vec beta_after = osqp_x.subvec(0, p - 1);
  double b_after = osqp_x(p);

  arma::vec margin_after = Y % (X * beta_after + b_after);
  arma::vec psi_value_after = psi(margin_after, psi_k);
  arma::vec psi_value_after_pos = psi_value_after % Y_pos_indicator;
  arma::vec psi_value_after_neg = psi_value_after % Y_neg_indicator;
  double fun_value_after = arma::sum(psi_value_after_pos) / n + gamma / 2 * arma::sum(arma::square(beta_after));

  int num_iteration = 1;

  while ((std::abs(fun_value_after - fun_value_before) / std::abs(fun_value_before) > max_rel_gap) && (num_iteration < max_iter_num))
  {
    beta_before = beta_after;
    b_before = b_after;
    fun_value_before = fun_value_after;
    margin_before = margin_after;

    indicator_u = (margin_before > 0) * 1.0;
    indicator = arma::conv_to<arma::vec>::from(indicator_u);
    c_value = arma::sum(b_vec % (1.0 - indicator));

    X_sub = X.rows(arma::find(indicator > 0));
    Y_sub = Y.elem(arma::find(indicator > 0));
    aug_X_sub = aug_X.rows(arma::find(indicator > 0));
    n_sub = X_sub.n_rows;

    // diagonal matrix P
    P_diag.set_size(p + n_sub + 1);
    P_diag.ones();
    P_diag.subvec(0, p - 1) *= n * gamma;
    P_diag.subvec(p, p + n_sub) *= 0;
    P = convert_to_diag_sp(P_diag);

    // linear vector q 
    q_u.set_size(p + n_sub + 1);
    q_u.ones();
    q_u.subvec(p + 1, p + n_sub) = (Y_sub > 0) * 1.0;
    q = arma::conv_to<arma::vec>::from(q_u);

    // coef matrix A
    A_lt = psi_k * (aug_X_sub.each_col() % Y_sub);
    A_rt = arma::eye(n_sub, n_sub);
    A_lm = arma::zeros(n_sub, p + 1);
    A_rm = A_rt;
    A_ll = arma::zeros(1, p + 1);
    A_rl = -b_vec.elem(arma::find(indicator > 0));

    A_dense = arma::join_cols(arma::join_rows(A_lt, A_rt), arma::join_rows(A_lm, A_rm), arma::join_rows(A_ll, A_rl));
    A = arma::conv_to<arma::sp_mat>::from(A_dense);

    // lower bound vector l
    l.set_size(2 * n_sub + 1);
    l.ones();
    l.subvec(n_sub, 2 * n_sub - 1) *= 0;
    l(2 * n_sub) = c_value - alpha * n_neg;

    // upper bound vector u
    u.set_size(2 * n_sub + 1);
    u.ones();
    u *= OSQP_INFTY;

    problem = osqpSetup_new(P, q, A, l, u);
    res = osqpSolve(problem);
    if (NumericVector::is_na(res[0]))
      stop("bug");

    osqp_x = as<NumericVector>(res[0]);
    beta_after = osqp_x.subvec(0, p - 1);
    b_after = osqp_x(p);

    margin_after = Y % (X * beta_after + b_after);
    psi_value_after = psi(margin_after, psi_k);
    psi_value_after_pos = psi_value_after % Y_pos_indicator;
    psi_value_after_neg = psi_value_after % Y_neg_indicator;
    fun_value_after = arma::sum(psi_value_after_pos) / n + gamma / 2 * arma::sum(arma::square(beta_after));

    num_iteration += 1;
  }

  return List::create(Named("beta") = beta_after,
                      Named("b") = b_after);
}