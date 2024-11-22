
import os
import subprocess

def minify_files(input_dir, file_type, output_dir):
    """Minify JavaScript and CSS files."""
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(file_type):
                input_file = os.path.join(root, file)
                output_file = os.path.join(output_dir, file.replace('.js', '.min.js') if file_type == '.js' else file.replace('.css', '.min.css'))
                if file_type == '.js':
                    subprocess.run(["uglifyjs", input_file, "-o", output_file])
                elif file_type == '.css':
                    subprocess.run(["csso", input_file, "--output", output_file])

if __name__ == "__main__":
    input_dir = "./static"
    output_dir = "./static/minified"
    os.makedirs(output_dir, exist_ok=True)
    minify_files(input_dir, ".js", output_dir)
    minify_files(input_dir, ".css", output_dir)
