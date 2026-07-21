namespace {

constexpr int kMaxNodes = 64;

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

extern "C" __global__ void particle_point_distances_kernel(
    const float* particles,
    const float* points,
    float* squared_distances,
    int* nearest_segments,
    int cable_count,
    int particle_count,
    int node_count,
    int point_count
) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = cable_count * particle_count * point_count;
    if (output_index >= total || node_count < 2 || node_count > kMaxNodes) {
        return;
    }

    const int point_index = output_index % point_count;
    const int particle_flat = output_index / point_count;
    const int particle_index = particle_flat % particle_count;
    const int cable_index = particle_flat / particle_count;
    const float* chain = particles + ((cable_index * particle_count + particle_index) * node_count * 3);
    const float px = points[3 * point_index];
    const float py = points[3 * point_index + 1];
    const float pz = points[3 * point_index + 2];

    float best_squared = 3.402823466e+38F;
    int best_segment = 0;
    for (int segment_index = 0; segment_index < node_count - 1; ++segment_index) {
        const float* start = chain + 3 * segment_index;
        const float* end = start + 3;
        const float sx = end[0] - start[0];
        const float sy = end[1] - start[1];
        const float sz = end[2] - start[2];
        const float length_sq = fmaxf(sx * sx + sy * sy + sz * sz, 1.0e-20f);
        const float dx = px - start[0];
        const float dy = py - start[1];
        const float dz = pz - start[2];
        const float position = fminf(fmaxf((dx * sx + dy * sy + dz * sz) / length_sq, 0.0f), 1.0f);
        const float rx = dx - position * sx;
        const float ry = dy - position * sy;
        const float rz = dz - position * sz;
        const float squared = rx * rx + ry * ry + rz * rz;
        if (squared < best_squared) {
            best_squared = squared;
            best_segment = segment_index;
        }
    }
    squared_distances[output_index] = best_squared;
    nearest_segments[output_index] = best_segment;
}

extern "C" __global__ void particle_support_distances_kernel(
    const float* particles,
    const float* points,
    float* squared_distances,
    int cable_count,
    int particle_count,
    int node_count,
    int point_count,
    int samples_per_segment
) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int segment_count = node_count - 1;
    const int samples_per_chain = segment_count * samples_per_segment;
    const int total = cable_count * particle_count * samples_per_chain;
    if (
        output_index >= total
        || node_count < 2
        || node_count > kMaxNodes
        || point_count < 1
        || samples_per_segment < 1
    ) {
        return;
    }

    const int chain_sample_index = output_index % samples_per_chain;
    const int particle_flat = output_index / samples_per_chain;
    const int particle_index = particle_flat % particle_count;
    const int cable_index = particle_flat / particle_count;
    const int segment_index = chain_sample_index / samples_per_segment;
    const int sample_index = chain_sample_index - segment_index * samples_per_segment;
    const float parameter = (static_cast<float>(sample_index) + 0.5f)
        / static_cast<float>(samples_per_segment);

    const float* chain = particles
        + ((cable_index * particle_count + particle_index) * node_count * 3);
    const float* start = chain + 3 * segment_index;
    const float* end = start + 3;
    const float sx = start[0] + parameter * (end[0] - start[0]);
    const float sy = start[1] + parameter * (end[1] - start[1]);
    const float sz = start[2] + parameter * (end[2] - start[2]);

    float best_squared = 3.402823466e+38F;
    for (int point_index = 0; point_index < point_count; ++point_index) {
        const float dx = sx - points[3 * point_index];
        const float dy = sy - points[3 * point_index + 1];
        const float dz = sz - points[3 * point_index + 2];
        best_squared = fminf(best_squared, dx * dx + dy * dy + dz * dz);
    }
    squared_distances[output_index] = best_squared;
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

// Observation gather status values. Keep these synchronized with cable_cuda.py.
// The raw XYZ value is retained whenever the ZED buffer contains finite values,
// even when the sample is rejected by range or confidence filtering.
extern "C" __global__ void gather_indexed_observations_kernel(
    const unsigned char* point_cloud,
    int point_step_bytes,
    const long long* pixel_indices,
    const unsigned char* confidence,
    int confidence_step_bytes,
    float* output,
    int* status,
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
    output[3 * output_index] = px;
    output[3 * output_index + 1] = py;
    output[3 * output_index + 2] = pz;

    if (!(isfinite(px) && isfinite(py) && isfinite(pz))) {
        status[output_index] = 1;
        return;
    }
    const float distance = sqrtf(px * px + py * py + pz * pz);
    if (!(distance >= depth_min && distance <= depth_max)) {
        status[output_index] = 2;
        return;
    }
    if (use_confidence != 0) {
        const float* confidence_row = reinterpret_cast<const float*>(confidence + y * confidence_step_bytes);
        const float value = confidence_row[x];
        if (!(isfinite(value) && value <= max_confidence)) {
            status[output_index] = 3;
            return;
        }
    }
    status[output_index] = 0;
}

extern "C" __global__ void radius_neighbor_inlier_kernel(
    const float* points,
    unsigned char* inlier,
    int point_count,
    float radius_squared,
    int minimum_neighbors
) {
    const int point_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (point_index >= point_count) {
        return;
    }
    const float px = points[3 * point_index];
    const float py = points[3 * point_index + 1];
    const float pz = points[3 * point_index + 2];
    int neighbor_count = 0;
    for (int other_index = 0; other_index < point_count; ++other_index) {
        if (other_index == point_index) {
            continue;
        }
        const float dx = px - points[3 * other_index];
        const float dy = py - points[3 * other_index + 1];
        const float dz = pz - points[3 * other_index + 2];
        if (dx * dx + dy * dy + dz * dz <= radius_squared) {
            ++neighbor_count;
            if (neighbor_count >= minimum_neighbors) {
                break;
            }
        }
    }
    inlier[point_index] = static_cast<unsigned char>(neighbor_count >= minimum_neighbors);
}
