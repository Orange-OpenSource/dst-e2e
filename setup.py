# Software Name : E2E-DST
# Version: 1.0.0
# SPDX-FileCopyrightText: Copyright (c) 2024 Orange
# SPDX-License-Identifier: MIT
# This software is distributed under the MIT,
# the text of which is available at https://opensource.org/licenses/MIT
# or see the "LICENSE" file for more details.
# Authors: Lucas Druart

from setuptools import setup, find_packages

MAILS = "lucas1.druart@orange.com"
DESC = "Package to perform completely neural Dialogue State Tracking (DST)."
URL = ""

setup(
        name = 'dstc11_e2e', 
        version = 0.1,
        author = "Lucas Druart",
        author_email = MAILS,
        keywords = DESC,
        url = URL,
        packages = find_packages(),
        include_package_data=True,
        zip_safe = False,
        install_requires =  [
            # Speechbrain is installed with git clone for develop branch
            # 'speechbrain==0.5.16', 
            'torch==2.0', 'torchaudio==2.0',
            'tensorboard==2.15.1', 'pandas==2.1.3',
            'transformers==4.36.0', 'openai-whisper==20231117',
            'seaborn==0.13.0',
            ],
        dependency_links = []
    )
