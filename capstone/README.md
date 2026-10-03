# Data Science and Machine Learning Vault

The required *Python* version for this project is *3.12.x.*

## About me

> Student should complete this section :)

## Setup environment

As usual setup your virtual environment:

```
$ python -m venv venv
$ source venv/bin/activate
$ pip install --upgrade pip setuptools
$ pip install -r requirements-dev.txt
$ pip install -e .
```

## Basic code compliance

```
$ black .
All done! ✨ 🍰 ✨
X files left unchanged.
$ mypy .
Success: no issues found in X source files
```

## About the CI/CD pipeline

This `monorepo` comes with a pre-configured CI/CD pipeline that is triggered every time a push is made to a **merge request** or when a **merge request** is integrated into the **main** branch.

The pipeline is configured to:

- Execute **code compliance** checks.
- Generate documentation.

> You may add more stages or jobs to the pipeline, but make sure you **do not remove the existing ones**. In addition, make sure you do your best to **keep the pipeline green** at all times.

## Information for students

When importing your `monorepo` for the first time, please make sure you keep the name of the project as `csds-352-machine-learning-vault` (**all lowercase**) and execute the following checklist:

- [ ] Write the **About me** section. Include your name, your email, and feel free to add any other information you want to share. It would be awesome if you can add your expectations for this course.
- [ ] The existing CI/CD pipeline is configured to generate a PDF slide deck intended to be used as a companion document for the course. Please, **update your name and email** in the `documentation/csds-352-vault.typ` file.
- [ ] Remove this section from the README file once you have completed the above tasks.
