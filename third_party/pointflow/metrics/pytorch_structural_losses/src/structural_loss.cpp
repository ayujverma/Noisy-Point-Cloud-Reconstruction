// src/structural_loss.cpp
// Patched for modern PyTorch 1.10+ / CUDA 11-12
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "src/approxmatch.cuh"
#include "src/nndistance.cuh"

#include <vector>
#include <iostream>

// Macros for input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

/*
input:
    set1 : batch_size * #dataset_points * 3
    set2 : batch_size * #query_points * 3
returns:
    match : batch_size * #query_points * #dataset_points
*/
std::vector<at::Tensor> ApproxMatch(at::Tensor set_d, at::Tensor set_q) {
    set_d = set_d.contiguous();
    set_q = set_q.contiguous();

    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m

    at::Tensor match = torch::empty({batch_size, n_query_points, n_dataset_points},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor temp = torch::empty({batch_size, (n_query_points + n_dataset_points)*2},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));

    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(match);
    CHECK_INPUT(temp);

    approxmatch(batch_size, n_dataset_points, n_query_points,
                set_d.data_ptr<float>(), set_q.data_ptr<float>(),
                match.data_ptr<float>(), temp.data_ptr<float>(),
                at::cuda::getCurrentCUDAStream());

    return {match, temp};
}

at::Tensor MatchCost(at::Tensor set_d, at::Tensor set_q, at::Tensor match) {
    set_d = set_d.contiguous();
    set_q = set_q.contiguous();
    match = match.contiguous();

    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m

    at::Tensor out = torch::empty({batch_size},
                                   torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));

    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(match);
    CHECK_INPUT(out);

    matchcost(batch_size, n_dataset_points, n_query_points,
              set_d.data_ptr<float>(), set_q.data_ptr<float>(),
              match.data_ptr<float>(), out.data_ptr<float>(),
              at::cuda::getCurrentCUDAStream());

    return out;
}

std::vector<at::Tensor> MatchCostGrad(at::Tensor set_d, at::Tensor set_q, at::Tensor match) {
    set_d = set_d.contiguous();
    set_q = set_q.contiguous();
    match = match.contiguous();

    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m

    at::Tensor grad1 = torch::empty({batch_size, n_dataset_points, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor grad2 = torch::empty({batch_size, n_query_points, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));

    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(match);
    CHECK_INPUT(grad1);
    CHECK_INPUT(grad2);

    matchcostgrad(batch_size, n_dataset_points, n_query_points,
                  set_d.data_ptr<float>(), set_q.data_ptr<float>(),
                  match.data_ptr<float>(), grad1.data_ptr<float>(), grad2.data_ptr<float>(),
                  at::cuda::getCurrentCUDAStream());

    return {grad1, grad2};
}

/*
input:
    set_d : batch_size * #dataset_points * 3
    set_q : batch_size * #query_points * 3
returns:
    dist1, idx1 : batch_size * #dataset_points
    dist2, idx2 : batch_size * #query_points
*/
std::vector<at::Tensor> NNDistance(at::Tensor set_d, at::Tensor set_q) {
    set_d = set_d.contiguous();
    set_q = set_q.contiguous();

    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m

    at::Tensor dist1 = torch::empty({batch_size, n_dataset_points},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor idx1 = torch::empty({batch_size, n_dataset_points},
                                   torch::TensorOptions().dtype(torch::kInt32).device(set_d.device()));
    at::Tensor dist2 = torch::empty({batch_size, n_query_points},
                                    torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor idx2 = torch::empty({batch_size, n_query_points},
                                   torch::TensorOptions().dtype(torch::kInt32).device(set_d.device()));

    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(dist1);
    CHECK_INPUT(idx1);
    CHECK_INPUT(dist2);
    CHECK_INPUT(idx2);

    nndistance(batch_size, n_dataset_points, set_d.data_ptr<float>(),
               n_query_points, set_q.data_ptr<float>(),
               dist1.data_ptr<float>(), idx1.data_ptr<int>(),
               dist2.data_ptr<float>(), idx2.data_ptr<int>(),
               at::cuda::getCurrentCUDAStream());

    return {dist1, idx1, dist2, idx2};
}

std::vector<at::Tensor> NNDistanceGrad(at::Tensor set_d, at::Tensor set_q,
                                       at::Tensor idx1, at::Tensor idx2,
                                       at::Tensor grad_dist1, at::Tensor grad_dist2) {
    set_d = set_d.contiguous();
    set_q = set_q.contiguous();
    idx1 = idx1.contiguous();
    idx2 = idx2.contiguous();
    grad_dist1 = grad_dist1.contiguous();
    grad_dist2 = grad_dist2.contiguous();

    int64_t batch_size = set_d.size(0);    
    int64_t n_dataset_points = set_d.size(1); // n
    int64_t n_query_points = set_q.size(1);   // m

    at::Tensor grad1 = torch::empty({batch_size, n_dataset_points, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));
    at::Tensor grad2 = torch::empty({batch_size, n_query_points, 3},
                                     torch::TensorOptions().dtype(torch::kFloat32).device(set_d.device()));

    CHECK_INPUT(set_d);
    CHECK_INPUT(set_q);
    CHECK_INPUT(idx1);
    CHECK_INPUT(idx2);
    CHECK_INPUT(grad_dist1);
    CHECK_INPUT(grad_dist2);
    CHECK_INPUT(grad1);
    CHECK_INPUT(grad2);

    nndistancegrad(batch_size, n_dataset_points, set_d.data_ptr<float>(),
                   n_query_points, set_q.data_ptr<float>(),
                   grad_dist1.data_ptr<float>(), idx1.data_ptr<int>(),
                   grad_dist2.data_ptr<float>(), idx2.data_ptr<int>(),
                   grad1.data_ptr<float>(), grad2.data_ptr<float>(),
                   at::cuda::getCurrentCUDAStream());

    return {grad1, grad2};
}

