"""The package imports and exposes a version."""


def test_package_imports_and_has_version():
    import propwash_gym

    assert isinstance(propwash_gym.__version__, str)
    assert propwash_gym.__version__ != ""
