Import('env', 'envCython', 'arch')

templates = Glob('#rednose/templates/*')

# TODO: get dependencies based on installation
sympy_helpers = "#rednose/helpers/sympy_helpers.py"
ekf_sym = "#rednose/helpers/ekf_sym.py"
ekf_sym_pyx = "#rednose/helpers/ekf_sym_pyx.pyx"
ekf_sym_cc = "#rednose/helpers/ekf_sym.cc"
common_ekf = "#rednose/helpers/common_ekf.cc"

to_build = {
    'live': ('examples/live_kf.py', 'examples/generated', True),
    'kinematic': ('examples/kinematic_kf.py', 'examples/generated', True),
    'compare': ('examples/test_compare.py', 'examples/generated', True),
    'pos_computer_4': ('#rednose/helpers/lst_sq_computer.py', 'examples/generated', False),
    'pos_computer_5': ('#rednose/helpers/lst_sq_computer.py', 'examples/generated', False),
    'feature_handler_5': ('#rednose/helpers/feature_handler.py', 'examples/generated', False),
}

found = {}

for target, (command, generated_folder, combined_lib) in to_build.items():
    if File(command).exists():
        found[target] = (command, generated_folder, combined_lib)

lib_target = [common_ekf]
for target, (command, generated_folder, combined_lib) in found.items():
    target_files = File([f'{generated_folder}/{target}.cpp', f'{generated_folder}/{target}.h'])
    command_file = File(command)

    env.Command(target_files,
                [templates, command_file, sympy_helpers, ekf_sym],
                command_file.get_abspath() + " " + target + " " + Dir(generated_folder).get_abspath())

    if combined_lib:
        lib_target.append(target_files[0])
    else:
        env.SharedLibrary(f'{generated_folder}/' + target, target_files[0])

libkf = env.SharedLibrary(f'{generated_folder}/libkf', lib_target)

lenv = envCython.Clone()
lenv["LINKFLAGS"] += [libkf[0].get_labspath()]
ekf_sym_so = lenv.Program('#rednose/helpers/ekf_sym_pyx.so', [ekf_sym_pyx, ekf_sym_cc, common_ekf])
lenv.Depends(ekf_sym_so, libkf)
