#!/usr/bin/env python

import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo

    return (mo,)


@app.cell
def __(mo):
    mo.md(r"""# Multimodal Harmful Meme Detection System""")
    return


@app.cell
def __(mo):
    image = mo.ui.file(filetypes=["image/*"], kind="area")
    image
    return (image,)


@app.cell
def __(image, mo):
    mo.stop(not len(image.value))
    mo.image(image.contents()).center()
    return


if __name__ == "__main__":
    app.run()
