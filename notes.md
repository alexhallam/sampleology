Notes on workflow

## Making git take fewer strokes

I used the following 
```
git config --global alias.ac '!git add -A && git commit'
```

This allows me to only have to run the following for a `git add -A` and `git commit -m`. A nice one-liner
```
git ac -m "message"
```

## Run tests

```
uv run pytest test_add.py -v --hypothesis-show-statistics
```

## Linting

```
uv run ruff check
```

## Fixing

```
uv run ruff check --fix
```

## Format

Make changes according to configuration. I do not like character limits less than 120.

```
uv run ruff format
```

## Automation with pre-commit

```
uv tool install pre-commit --with pre-commit-uv
pre-commit install # this makes it so that checks run on every git commit
```

Now I can do:

```
git add -A
pre-commit run --all-files
```



`uv` - package management

`ruff` - linting

`chex` - for writing reliable jax code

`hypothesis` - fancy testing? (hypothesis)[https://hypothesis.readthedocs.io/en/latest/tutorial/introduction.html]



