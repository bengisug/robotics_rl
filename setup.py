from setuptools import setup, find_packages

setup(
    name='robotics_rl',
    version='0.1',
    url='',
    license='MIT',
    author='Bengisu Guresti, Tolga Ok, Asil Can Yilmaz',
    author_email='bengisuguresti0@gmail.com, tolgaokk@gmail.com, asilcy2015@gmail.com',
    description='Robot manipulation training with Deep Reinforcement Learning',
      
    # Package info
    packages=find_packages(),
    install_requires=["gym==0.10.5",
                      "pybullet==2.6.6",
                      "matplotlib==3.2.1",
                      "numpy==1.18.1",
                      "tensorboard==1.15.0",
                      "tensorboardX==2.0",
                      "torch==1.4.0",
                      "future==0.18.2",
                      "PyYAML==5.3.1",
                      ],
    zip_safe=False
)
