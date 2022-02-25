from setuptools import setup

setup(name='gds_phuncs',
	version='0.1',
	description='module for writing photonic device cad using phidl',
	url='https://github.com/mudyeh/gds_phuncs',
	author='Matthew Yeh',
	author_email='myeh@g.harvard.edu',
	license='MIT',
	packages=['gds_phuncs'],
	install_requires=[
		'numpy',
		'pandas',
		'scipy',
		'lmfit',
		'matplotlib',
		'gdspy',
		'phidl'
		],
	zip_safe=False)
