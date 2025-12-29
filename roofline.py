import matplotlib.pyplot as plt
import numpy as np

def generate_roofline_chart():
    # --- 1. Hardware Specifications ---
    # GPU: AMD MI210 (Vector)
    gpu_peak_flops = 45.3 * 10**12   # 45.3 TFLOPs
    gpu_bw = 1.6 * 10**12            # 1.6 TB/s

    # CPU: EPYC 7V13 (User Specs)
    cpu_peak_flops = 64*2.5*32*2*10**9   # 10.24 TFLOPs
    cpu_bw = 204.8 * 10**9           # 204.8 GB/s

    # --- 2. Algorithm Intensities ---
    # Distance Kernel: 9 FLOPs / 16 Bytes
    ai_distance = 0.5625

    # Sorting Kernel: 1 Op / 32 Bytes
    ai_sorting = 0.03125

    # --- 3. Setup Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # X-axis: Operational Intensity (log scale range)
    x = np.logspace(-2.5, 2.5, 1000)

    # --- 4. Plot GPU Roofline ---
    gpu_mem_bound = x * gpu_bw
    gpu_compute_bound = np.full_like(x, gpu_peak_flops)
    gpu_roof = np.minimum(gpu_mem_bound, gpu_compute_bound)

    ax.loglog(x, gpu_roof, 'r-', linewidth=2.5, label='GPU: AMD MI210 (Vector)')

    # GPU Ridge Point (Threshold)
    gpu_ridge = gpu_peak_flops / gpu_bw
    ax.plot(gpu_ridge, gpu_peak_flops, 'ro', markersize=8)
    ax.text(gpu_ridge, gpu_peak_flops * 1.3, f'GPU Ridge\n{gpu_ridge:.1f} FLOPs/Byte',
            color='red', ha='center', fontweight='bold')

    # --- 5. Plot CPU Roofline ---
    cpu_mem_bound = x * cpu_bw
    cpu_compute_bound = np.full_like(x, cpu_peak_flops)
    cpu_roof = np.minimum(cpu_mem_bound, cpu_compute_bound)

    ax.loglog(x, cpu_roof, 'b-', linewidth=2.5, label='CPU: EPYC 7V13 (2X64-core)')

    # CPU Ridge Point (Threshold)
    cpu_ridge = cpu_peak_flops / cpu_bw
    ax.plot(cpu_ridge, cpu_peak_flops, 'bo', markersize=8)
    ax.text(cpu_ridge, cpu_peak_flops * 1.3, f'CPU Ridge\n{cpu_ridge:.2f} FLOPs/Byte',
            color='blue', ha='center', fontweight='bold')

    # --- 6. Plot Algorithm Points (Performance = AI * Bandwidth) ---
    # Distance Kernel Points
    gpu_perf_dist = ai_distance * gpu_bw
    cpu_perf_dist = ai_distance * cpu_bw

    ax.plot(ai_distance, gpu_perf_dist, 'r*', markersize=18, label='Distance Algo (GPU)')
    ax.plot(ai_distance, cpu_perf_dist, 'b*', markersize=18, label='Distance Algo (CPU)')

    # Sorting Kernel Points
    gpu_perf_sort = ai_sorting * gpu_bw
    cpu_perf_sort = ai_sorting * cpu_bw

    ax.plot(ai_sorting, gpu_perf_sort, 'rx', markersize=14, markeredgewidth=3, label='Sorting Algo (GPU)')
    ax.plot(ai_sorting, cpu_perf_sort, 'bx', markersize=14, markeredgewidth=3, label='Sorting Algo (CPU)')

    # --- 7. Annotations & Styling ---
    # Annotate Distance AI
    ax.annotate(f'Distance AI: {ai_distance:.2f}',
                xy=(ai_distance, gpu_perf_dist),
                xytext=(ai_distance * 1.5, gpu_perf_dist / 3),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Annotate Sorting AI
    ax.annotate(f'Sorting AI: {ai_sorting:.2f}(Extreme Mem Bound)',
                xy=(ai_sorting, gpu_perf_sort),
                xytext=(ai_sorting / 5, gpu_perf_sort * 8),
                arrowprops=dict(facecolor='black', shrink=0.05))

    # Bandwidth Labels
    ax.text(0.005, gpu_bw * 1.1, f'GPU BW: 1.6 TB/s', color='red', fontsize=10)
    ax.text(0.005, cpu_bw * 1.1, f'CPU BW: 205 GB/s', color='blue', fontsize=10)

    ax.set_xlabel('Operational Intensity (FLOPs/Byte)', fontsize=12)
    ax.set_ylabel('Performance (FLOPs/sec)', fontsize=12)
    ax.set_title('Roofline Model: Distance vs Sorting Algorithms', fontsize=16)
    ax.grid(True, which="both", ls="-", alpha=0.4)
    ax.legend(loc='lower right')

    # Save and Show
    plt.tight_layout()
    plt.savefig('final_roofline.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    generate_roofline_chart()
