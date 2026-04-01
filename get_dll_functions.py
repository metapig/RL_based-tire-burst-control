import pefile

dll_path = r"D:\CarSim2023\CarSim2023.2_Prog\Programs\solvers\carsim_64.dll"

pe = pefile.PE(dll_path)

print("=== Exported Functions ===")
for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
    if exp.name:
        print(exp.name.decode())