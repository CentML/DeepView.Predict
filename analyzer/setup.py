import codecs
import os
import pkg_resources
import re
import sysconfig
import subprocess

from setuptools import setup, find_packages
from setuptools.command.build import build

# Acknowledgement: This setup.py was adapted from Hynek Schlawack's Python
#                  Packaging Guide
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty

###################################################################

NAME = "deepview-predict"
PACKAGES = find_packages()
META_PATH = os.path.join("habitat", "__init__.py")
README_PATH = "../README.md"
PYTHON_REQUIRES = ">=3.7"

PYTHON_VERSION = sysconfig.get_python_version().replace('.', '')

SETUP_REQUIRES = [
    "patchelf",
    "incremental"
]

PACKAGE_DATA = {
    "habitat": [
        "analysis/mlp/devices.csv",
        "data/hints.yml",
        "data/devices.yml",
        "data/bmm/model.pth",
        "data/conv_transpose2d/model.pth",
        "data/conv2d/model.pth",
        "data/kernels.sqlite",
        "data/linear/model.pth",
        "data/lstm/model.pth",
        "habitat_cuda.cpython-{}*.so".format(PYTHON_VERSION),
    ],
}

INSTALL_REQUIRES = [
    "pyyaml",
    "torch>=1.4.0",
    "pandas>=1.1.2",
    "tqdm>=4.49.0",
    "nvidia-cuda-cupti-cu11==11.7.101",
    "nvidia-cuda-runtime-cu11==11.7.99",
    "incremental"
]

KEYWORDS = [
    "neural networks",
    "pytorch",
    "performance",
    "profiler",
    "predictions",
]

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3 :: Only",
]

class CustomBuildCommand(build):
    def run(self):
        # Need to update the rpath of the habitat_cuda.cpython library
        # Ensures that it links to the libraries included in the wheel
        patchelf_bin_path = pkg_resources.get_distribution("patchelf").location + "/EGG-INFO/scripts/patchelf"
        habitat_dir = os.listdir("habitat")
        curr_python_ver = "{}".format(PYTHON_VERSION)
        library_name = ""
        for fname in habitat_dir:
            if fname.startswith("habitat_cuda.cpython-"+curr_python_ver) and fname.endswith(".so"):
                library_name = fname
                break
        
        habitat_library = "habitat/"+library_name
        # Set rpath to the SO files found in the pip package
        cmd = [patchelf_bin_path, '--print-rpath', habitat_library]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        original_rpath = proc.stdout.read().strip()
        package_rpath = "$ORIGIN/../nvidia/cuda_runtime/lib:$ORIGIN/../nvidia/cuda_cupti/lib"
        cmd = [patchelf_bin_path, '--set-rpath', package_rpath, habitat_library]
        subprocess.check_call(cmd)

        build.run(self)

        cmd = [patchelf_bin_path, '--set-rpath', original_rpath, habitat_library]
        subprocess.check_call(cmd)


###################################################################

HERE = os.path.abspath(os.path.dirname(__file__))


def read(*parts):
    """
    Build an absolute path from *parts* and return the contents of the
    resulting file. Assume UTF-8 encoding.
    """
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


META_FILE = read(META_PATH)


def find_meta(meta):
    """
    Extract __*meta*__ from META_FILE.
    """
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta),
        META_FILE, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        description=find_meta("description"),
        license=find_meta("license"),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        long_description=read(README_PATH),
        use_incremental=True,
        long_description_content_type="text/markdown",
        cmdclass= {
            "build": CustomBuildCommand
        },
        packages=PACKAGES,
        package_data=PACKAGE_DATA,
        python_requires=PYTHON_REQUIRES,
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        keywords=KEYWORDS,
        options={
            "bdist_wheel": {
                "python_tag": "py"+ PYTHON_VERSION
            }
        }
    )
