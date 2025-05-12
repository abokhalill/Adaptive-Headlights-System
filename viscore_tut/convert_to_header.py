import os

def convert_to_header(input_file, output_file, variable_name):
    with open(input_file, "rb") as f:
        model_data = f.read()

    with open(output_file, "w") as f:
        f.write(f"#ifndef {variable_name.upper()}_H\n")
        f.write(f"#define {variable_name.upper()}_H\n\n")
        f.write(f"const unsigned char {variable_name}[] = {{\n")

        for i, byte in enumerate(model_data):
            if i % 12 == 0:
                f.write("\n    ")
            f.write(f"0x{byte:02X}, ")

        f.write("\n};\n\n")
        f.write(f"const unsigned int {variable_name}_len = {len(model_data)};\n\n")
        f.write("#endif // " + f"{variable_name.upper()}_H\n")

if __name__ == "__main__":
    input_file = "brightness_model.tflite"  # Replace with your .tflite file path
    output_file = "brightness_model.h"  # Replace with your desired .h file name
    variable_name = "brightness_model"  # Replace with your desired variable name

    if not os.path.exists(input_file):
        print(f"Error: {input_file} does not exist.")
    else:
        convert_to_header(input_file, output_file, variable_name)
        print(f"Header file {output_file} created successfully.")
