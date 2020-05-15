Import('env', 'arch')

templates = Glob('#rednose/templates/*')

# TODO: get dependencies based on installation
sympy_helpers = "#rednose/helpers/sympy_helpers.py"
ekf_sym = "#rednose/helpers/ekf_sym.py"

to_build = {
    'live': ('examples/live_kf.py', 'examples/generated'),
    'kinematic': ('examples/kinematic_kf.py', 'examples/generated'),
    'pos_computer_4': ('rednose/helpers/lst_sq_computer.py', 'examples/generated'),
    'pos_computer_5': ('rednose/helpers/lst_sq_computer.py', 'examples/generated'),
    'feature_handler_5': ('rednose/helpers/feature_handler.py', 'examples/generated'),
}


found = {}

for target, (command, generated_folder) in to_build.items():
    if File(command).exists():
        found[target] = (command, generated_folder)

for target, (command, generated_folder) in found.items():
    target_files = File([f'{generated_folder}/{target}.cpp', f'{generated_folder}/{target}.h'])
    command_file = File(command)

    env.Command(target_files,
                [templates, command_file, sympy_helpers, ekf_sym],
                command_file.get_abspath() + " " + target + " " + Dir(generated_folder).get_abspath())

    env.SharedLibrary(f'{generated_folder}/' + target, target_files[0])
