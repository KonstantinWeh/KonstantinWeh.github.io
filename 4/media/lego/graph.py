import matplotlib.pyplot as plt

psnr_values = [0.0]
with open("media/lego/validation_psnr_raw.txt", "r") as f:
    for line in f:
        try:
            val = float(line.strip())
            psnr_values.append(val)
        except ValueError:
            continue

iters = [0] + list(range(2000, 20000 + 1, 2000))

plt.figure(figsize=(8, 5))
plt.plot(iters[:len(psnr_values)], psnr_values, marker="o")
plt.xlabel("Iteration")
plt.ylabel("PSNR")
plt.title("Validation PSNR vs Iterations")
plt.grid(True)
plt.savefig("media/lego/psnr_curve_lego.jpg")
plt.tight_layout()
plt.show()
