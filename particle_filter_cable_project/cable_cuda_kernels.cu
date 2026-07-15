namespace {

constexpr int kMaxNodes = 64;
constexpr int kMaxAnchors = 16;

__device__ inline void normalize3(float& x, float& y, float& z) {
    const float norm_sq = x * x + y * y + z * z;
    if (norm_sq <= 1.0e-20f) {
        x = 1.0f;
        y = 0.0f;
        z = 0.0f;
        return;
    }
    const float inverse = rsqrtf(norm_sq);
    x *= inverse;
    y *= inverse;
    z *= inverse;
}

__device__ inline void constrain_one_chain(
    float* chain,
    const float* endpoints,
    int node_count,
    float segment_length,
    int iterations,
    float tolerance
) {
    const int last = node_count - 1;
    chain[0] = endpoints[0];
    chain[1] = endpoints[1];
    chain[2] = endpoints[2];
    chain[3 * last] = endpoints[3];
    chain[3 * last + 1] = endpoints[4];
    chain[3 * last + 2] = endpoints[5];

    for (int iteration = 0; iteration < iterations; ++iteration) {
        chain[3 * last] = endpoints[3];
        chain[3 * last + 1] = endpoints[4];
        chain[3 * last + 2] = endpoints[5];
        for (int index = last - 1; index >= 0; --index) {
            float dx = chain[3 * index] - chain[3 * (index + 1)];
            float dy = chain[3 * index + 1] - chain[3 * (index + 1) + 1];
            float dz = chain[3 * index + 2] - chain[3 * (index + 1) + 2];
            normalize3(dx, dy, dz);
            chain[3 * index] = chain[3 * (index + 1)] + segment_length * dx;
            chain[3 * index + 1] = chain[3 * (index + 1) + 1] + segment_length * dy;
            chain[3 * index + 2] = chain[3 * (index + 1) + 2] + segment_length * dz;
        }

        chain[0] = endpoints[0];
        chain[1] = endpoints[1];
        chain[2] = endpoints[2];
        for (int index = 0; index < last; ++index) {
            float dx = chain[3 * (index + 1)] - chain[3 * index];
            float dy = chain[3 * (index + 1) + 1] - chain[3 * index + 1];
            float dz = chain[3 * (index + 1) + 2] - chain[3 * index + 2];
            normalize3(dx, dy, dz);
            chain[3 * (index + 1)] = chain[3 * index] + segment_length * dx;
            chain[3 * (index + 1) + 1] = chain[3 * index + 1] + segment_length * dy;
            chain[3 * (index + 1) + 2] = chain[3 * index + 2] + segment_length * dz;
        }
        const float ex = chain[3 * last] - endpoints[3];
        const float ey = chain[3 * last + 1] - endpoints[4];
        const float ez = chain[3 * last + 2] - endpoints[5];
        if (sqrtf(ex * ex + ey * ey + ez * ez) <= tolerance) {
            break;
        }
    }
}

}  // namespace

extern "C" __global__ void constrain_chains_kernel(
    float* chains,
    const float* endpoints,
    int chain_count,
    int node_count,
    float segment_length,
    int iterations,
    float tolerance
) {
    const int chain_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (chain_index >= chain_count || node_count < 2 || node_count > kMaxNodes) {
        return;
    }
    constrain_one_chain(
        chains + chain_index * node_count * 3,
        endpoints,
        node_count,
        segment_length,
        iterations,
        tolerance
    );
}

extern "C" __global__ void build_ransac_chains_kernel(
    const float* anchors,
    const float* endpoints,
    float* chains,
    int hypothesis_count,
    int anchor_count,
    int node_count,
    float segment_length,
    int constraint_iterations,
    float constraint_tolerance
) {
    const int hypothesis = blockIdx.x * blockDim.x + threadIdx.x;
    if (
        hypothesis >= hypothesis_count || node_count < 2 || node_count > kMaxNodes ||
        anchor_count < 1 || anchor_count > kMaxAnchors
    ) {
        return;
    }

    float polyline[(kMaxAnchors + 2) * 3];
    float cumulative[kMaxAnchors + 2];
    const int polyline_count = anchor_count + 2;
    polyline[0] = endpoints[0];
    polyline[1] = endpoints[1];
    polyline[2] = endpoints[2];
    for (int index = 0; index < anchor_count; ++index) {
        const int source = (hypothesis * anchor_count + index) * 3;
        polyline[3 * (index + 1)] = anchors[source];
        polyline[3 * (index + 1) + 1] = anchors[source + 1];
        polyline[3 * (index + 1) + 2] = anchors[source + 2];
    }
    polyline[3 * (polyline_count - 1)] = endpoints[3];
    polyline[3 * (polyline_count - 1) + 1] = endpoints[4];
    polyline[3 * (polyline_count - 1) + 2] = endpoints[5];

    cumulative[0] = 0.0f;
    for (int index = 1; index < polyline_count; ++index) {
        const float dx = polyline[3 * index] - polyline[3 * (index - 1)];
        const float dy = polyline[3 * index + 1] - polyline[3 * (index - 1) + 1];
        const float dz = polyline[3 * index + 2] - polyline[3 * (index - 1) + 2];
        cumulative[index] = cumulative[index - 1] + sqrtf(dx * dx + dy * dy + dz * dz);
    }

    float resampled[kMaxNodes * 3];
    const float total = cumulative[polyline_count - 1];
    for (int node = 0; node < node_count; ++node) {
        const float target = total * static_cast<float>(node) / static_cast<float>(node_count - 1);
        int segment = 0;
        while (segment + 1 < polyline_count - 1 && cumulative[segment + 1] < target) {
            ++segment;
        }
        const float span = fmaxf(cumulative[segment + 1] - cumulative[segment], 1.0e-12f);
        const float alpha = fminf(fmaxf((target - cumulative[segment]) / span, 0.0f), 1.0f);
        for (int axis = 0; axis < 3; ++axis) {
            resampled[3 * node + axis] =
                polyline[3 * segment + axis] +
                alpha * (polyline[3 * (segment + 1) + axis] - polyline[3 * segment + axis]);
        }
    }

    float* chain = chains + hypothesis * node_count * 3;
    chain[0] = resampled[0];
    chain[1] = resampled[1];
    chain[2] = resampled[2];
    for (int node = 1; node < node_count; ++node) {
        float dx = resampled[3 * node] - resampled[3 * (node - 1)];
        float dy = resampled[3 * node + 1] - resampled[3 * (node - 1) + 1];
        float dz = resampled[3 * node + 2] - resampled[3 * (node - 1) + 2];
        normalize3(dx, dy, dz);
        chain[3 * node] = chain[3 * (node - 1)] + segment_length * dx;
        chain[3 * node + 1] = chain[3 * (node - 1) + 1] + segment_length * dy;
        chain[3 * node + 2] = chain[3 * (node - 1) + 2] + segment_length * dz;
    }
    constrain_one_chain(
        chain,
        endpoints,
        node_count,
        segment_length,
        constraint_iterations,
        constraint_tolerance
    );
}

extern "C" __global__ void gather_indexed_points_kernel(
    const unsigned char* point_cloud,
    int point_step_bytes,
    const long long* pixel_indices,
    const unsigned char* confidence,
    int confidence_step_bytes,
    float* output,
    int width,
    int candidate_count,
    float depth_min,
    float depth_max,
    float max_confidence,
    int use_confidence
) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (output_index >= candidate_count) {
        return;
    }
    const long long pixel = pixel_indices[output_index];
    const int y = pixel / width;
    const int x = pixel - y * width;
    const float* point = reinterpret_cast<const float*>(point_cloud + y * point_step_bytes) + 4 * x;
    const float px = point[0];
    const float py = point[1];
    const float pz = point[2];
    const float distance = sqrtf(px * px + py * py + pz * pz);
    bool valid = isfinite(px) && isfinite(py) && isfinite(pz);
    valid = valid && distance >= depth_min && distance <= depth_max;
    if (use_confidence != 0) {
        const float* confidence_row = reinterpret_cast<const float*>(confidence + y * confidence_step_bytes);
        const float value = confidence_row[x];
        valid = valid && isfinite(value) && value <= max_confidence;
    }
    if (valid) {
        output[3 * output_index] = px;
        output[3 * output_index + 1] = py;
        output[3 * output_index + 2] = pz;
    } else {
        output[3 * output_index] = nanf("");
        output[3 * output_index + 1] = nanf("");
        output[3 * output_index + 2] = nanf("");
    }
}
