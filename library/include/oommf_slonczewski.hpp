#ifndef RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#define RUNTIME_INCLUDE_SYCL_SYCL_HPP_
#include <CL/sycl.hpp>
namespace sycl = cl::sycl;

#endif // RUNTIME_INCLUDE_SYCL_SYCL_HPP_

#include "amul.hpp"
#include "constants.hpp"

// Original implementation by Mykola Dvornik for mumax2
// Modified for mumax3 by Arne Vansteenkiste, 2013, 2016

template <typename dataT>
class addoommfslonczewskitorque_kernel {
	public:
	    using read_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read, sycl::access::target::global_buffer>;
	    using write_accessor =
		    sycl::accessor<dataT, 1, sycl::access::mode::read_write, sycl::access::target::global_buffer>;
		addoommfslonczewskitorque_kernel(write_accessor tXPtr, write_accessor tYPtr, write_accessor tZPtr,
		             read_accessor mxPtr, read_accessor myPtr, read_accessor mzPtr,
		             read_accessor MsPtr, dataT Ms_mul,
					 read_accessor jzPtr, dataT jz_mul,
					 read_accessor pxPtr, dataT px_mul,
					 read_accessor pyPtr, dataT py_mul,
					 read_accessor pzPtr, dataT pz_mul,
					 read_accessor alphaPtr, dataT alpha_mul,
					 read_accessor pfixPtr, dataT pfix_mul,
					 read_accessor pfreePtr, dataT pfree_mul,
					 read_accessor lambdafixPtr, dataT lambdafix_mul,
					 read_accessor lambdafreePtr, dataT lambdafree_mul,
					 read_accessor epsPrimePtr, dataT epsPrime_mul,
					 read_accessor fltPtr, dataT flt_mul,
					 size_t N)
		    :	tx(tXPtr),
				ty(tYPtr),
				tz(tZPtr),
				mx_(mxPtr),
				my_(myPtr),
				mz_(mzPtr),
				Ms_(MsPtr),
				Ms_mul(Ms_mul),
				jz_(jzPtr),
				jz_mul(jz_mul),
				px_(pxPtr),
				px_mul(px_mul),
				py_(pyPtr),
				py_mul(py_mul),
				pz_(pzPtr),
				pz_mul(pz_mul),
				alpha_(alphaPtr),
				alpha_mul(alpha_mul),
				pfix_(pfixPtr),
				pfix_mul(pfix_mul),
				pfree_(pfreePtr),
				pfree_mul(pfree_mul),
				lambdafix_(lambdafixPtr),
				lambdafix_mul(lambdafix_mul),
				lambdafree_(lambdafreePtr),
				lambdafree_mul(lambdafree_mul),
				epsPrime_(epsPrimePtr),
				epsPrime_mul(epsPrime_mul),
				flt_(fltPtr),
				flt_mul(flt_mul),
				N(N) {}
		void operator()(sycl::nd_item<1> item) {
			size_t stride = item.get_global_range(0);
			for (size_t gid = item.get_global_linear_id(); gid < N; gid += stride) {
				dataT  J = amul(jz_, jz_mul, gid);
				dataT  Ms           = amul(Ms_, Ms_mul, i);
				if ((J == (dataT)(0.0)) || (Ms == (dataT)(0.0))) {
					return;
				}

				dataT  alpha        = amul(alpha_, alpha_mul, gid);
				dataT  flt          = amul(flt_, flt_mul, gid);
				dataT  pfix         = amul(pfix_, pfix_mul, gid);
				dataT  pfree        = amul(pfree_, pfree_mul, gid);
				dataT  lambdafix    = amul(lambdafix_, lambdafix_mul, gid);
				dataT  lambdafree   = amul(lambdafree_, lambdafree_mul, gid);
				dataT  epsilonPrime = amul(epsPrime_, epsPrime_mul, gid);

				dataT px = amul(px_, px_mul, gid);
				dataT py = amul(py_, py_mul, gid);
				dataT pz = amul(pz_, pz_mul, gid);

				dataT pnormFac = sycl::rsqrt((px*px) + (py*py) + (pz*pz));
				
				px *= pnormFac; py *= pnormFac; pz *= pnormFac;

				dataT beta    = (HBAR / QE) * (J / ((dataT)(2.0)*flt*Ms) );
				dataT lambdafix2 = lambdafix * lambdafix;
				dataT lambdafree2 = lambdafree * lambdafree;
				dataT lambdafreePlus = sqrt(lambdafree2 + (dataT)(1.0));
				dataT lambdafixPlus = sqrt(lambdafix2 + (dataT)(1.0));
				dataT lambdafreeMinus = sqrt(lambdafree2 - (dataT)(1.0));
				dataT lambdafixMinus = sqrt(lambdafix2 - (dataT)(1.0));
				dataT plus_ratio = lambdafreePlus / lambdafixPlus;
				dataT minus_ratio = (dataT)(1.0);
				if (lambdafreeMinus > (dataT)(0.0)) {
					minus_ratio = lambdafixMinus / lambdafreeMinus;
				}
				// Compute q_plus and q_minus
				dataT plus_factor = pfix * lambdafix2 * plus_ratio;
				dataT minus_factor = pfree * lambdafree2 * minus_ratio;
				dataT q_plus = plus_factor + minus_factor;
				dataT q_minus = plus_factor - minus_factor;
				dataT lplus2 = lambdafreePlus * lambdafixPlus;
				dataT lminus2 = lambdafreeMinus * lambdafixMinus;
				dataT pdotm = dot(p, m);
				dataT A_plus = lplus2 + (lminus2 * pdotm);
				dataT A_minus = lplus2 - (lminus2 * pdotm);
				dataT epsilon = (q_plus / A_plus) - (q_minus / A_minus);

				dataT A = beta * epsilon;
				dataT B = beta * epsilonPrime;

				dataT gilb     = (dataT)(1.0) / ((dataT)(1.0) + alpha * alpha);
				dataT mxpxmFac = gilb * (A + alpha * B);
				dataT pxmFac   = gilb * (B - alpha * A);

				dataT mx = mx_[gid]; dataT my = my_[gid]; dataT mz = mz_[gid];

				dataT c0 = py * mz; dataT d0 = my * pz;
				dataT c1 = px * mz; dataT d1 = mx * pz;
				dataT c2 = px * my; dataT d2 = mx * py;
				dataT pxmx = c0 - d0;
				dataT pxmy = d1 - c1;
				dataT pxmz = c2 - d2;

				c0 = my * pxmz; d0 = pxmy * mz;
				c1 = mx * pxmz; d1 = pxmx * mz;
				c2 = mx * pxmy; d2 = pxmx * my;

				dataT m2x = c0 - d0;
				dataT m2y = d1 - c1;
				dataT m2z = c2 - d2;

				tx[gid] += mxpxmFac * m2x + pxmFac * pxmx;
				ty[gid] += mxpxmFac * m2y + pxmFac * pxmy;
				tz[gid] += mxpxmFac * m2z + pxmFac * pxmz;
			}
		}
	private:
	    write_accessor tx;
	    write_accessor ty;
	    write_accessor tz;
	    read_accessor  mx_;
	    read_accessor  my_;
	    read_accessor  mz_;
	    read_accessor  jz_;
		dataT          jz_mul;
	    read_accessor  px_;
		dataT          px_mul;
	    read_accessor  py_;
		dataT          py_mul;
	    read_accessor  pz_;
		dataT          jz_mul;
		read_accessor  alpha_;
		dataT          alpha_mul;
	    read_accessor  pfix_;
		dataT          pfix_mul;
	    read_accessor  pfree_;
		dataT          pfree_mul;
	    read_accessor  lambdafix_;
		dataT          lambdafix_mul;
	    read_accessor  lambdafree_;
		dataT          lambdafree_mul;
	    read_accessor  epsPrime_;
		dataT          epsPrime_mul;
	    read_accessor  flt_;
		dataT          flt_mul;
		size_t         N;
};

template <typename dataT>
void addoommfslonczewskitorque_async(sycl::queue funcQueue,
                 sycl::buffer<dataT, 1> *tx, sycl::buffer<dataT, 1> *ty, sycl::buffer<dataT, 1> *tz,
                 sycl::buffer<dataT, 1> *mx, sycl::buffer<dataT, 1> *my, sycl::buffer<dataT, 1> *mz,
                 sycl::buffer<dataT, 1> *Ms, dataT Ms_mul,
                 sycl::buffer<dataT, 1> *jz, dataT jz_mul,
                 sycl::buffer<dataT, 1> *px, dataT px_mul,
                 sycl::buffer<dataT, 1> *py, dataT py_mul,
				 sycl::buffer<dataT, 1> *pz, dataT pz_mul,
                 sycl::buffer<dataT, 1> *alpha, dataT alpha_mul,
                 sycl::buffer<dataT, 1> *pfix, dataT pfix_mul,
                 sycl::buffer<dataT, 1> *pfree, dataT pfree_mul,
                 sycl::buffer<dataT, 1> *lambdafix, dataT lambdafix_mul,
                 sycl::buffer<dataT, 1> *lambdafree, dataT lambdafree_mul,
                 sycl::buffer<dataT, 1> *epsPrime, dataT epsPrime_mul,
                 sycl::buffer<dataT, 1> *flt, dataT flt_mul,
                 size_t N,
				 size_t gsize,
				 size_t lsize) {

    funcQueue.submit([&] (sycl::handler& cgh) {
        auto tx_acc = tx->template get_access<sycl::access::mode::read_write>(cgh);
        auto ty_acc = ty->template get_access<sycl::access::mode::read_write>(cgh);
        auto tz_acc = tz->template get_access<sycl::access::mode::read_write>(cgh);
        auto mx_acc = mx->template get_access<sycl::access::mode::read>(cgh);
        auto my_acc = my->template get_access<sycl::access::mode::read>(cgh);
        auto mz_acc = mz->template get_access<sycl::access::mode::read>(cgh);
        auto Ms_acc = Ms->template get_access<sycl::access::mode::read>(cgh);
        auto jz_acc = jz->template get_access<sycl::access::mode::read>(cgh);
        auto px_acc = px->template get_access<sycl::access::mode::read>(cgh);
        auto py_acc = py->template get_access<sycl::access::mode::read>(cgh);
        auto pz_acc = pz->template get_access<sycl::access::mode::read>(cgh);
        auto alpha_acc = alpha_->template get_access<sycl::access::mode::read>(cgh);
        auto pfix_acc = pfix->template get_access<sycl::access::mode::read>(cgh);
        auto pfree_acc = pfree->template get_access<sycl::access::mode::read>(cgh);
        auto lambdafix_acc = lambdafix->template get_access<sycl::access::mode::read>(cgh);
        auto lambdafree_acc = lambdafree->template get_access<sycl::access::mode::read>(cgh);
        auto epsPrime_acc = epsPrime->template get_access<sycl::access::mode::read>(cgh);
        auto flt_acc = flt->template get_access<sycl::access::mode::read>(cgh);

        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(gsize),
		                                   sycl::range<1>(lsize)),
		                 addoommfslonczewskitorque_kernel<dataT>(tx_acc, ty_acc, tz_acc,
						                           mx_acc, my_acc, mz_acc,
												   Ms_acc, Ms_mul,
												   jz_acc, jz_mul,
												   px_acc, px_mul,
												   py_acc, py_mul,
												   pz_acc, pz_mul,
												   alpha_acc, alpha_mul,
												   pfix_acc, pfix_mul,
												   pfree_acc, pfree_mul,
												   lambdafix_acc, lambdafix_mul,
												   lambdafree_acc, lambdafree_mul,
												   epsPrime_acc, epsPrime_mul,
												   flt_acc, flt_mul,
												   N));
    });
}
