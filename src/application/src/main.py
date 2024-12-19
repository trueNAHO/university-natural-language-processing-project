#!/usr/bin/env python

import marimo

__generated_with = "0.9.14"
app = marimo.App()


@app.cell
def __():
    import marimo as mo
    import tempfile

    from use import main as evaluate

    return evaluate, mo, tempfile


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
def __(image, mo, tempfile):
    mo.stop(not len(image.value))

    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as file:
        file.write(image.contents())
        image_file = file.name

    mo.image(image.contents()).center()
    return file, image_file


@app.cell
def __(mo):
    image_text = mo.ui.text_area(full_width=True, label="Text written on image.")
    image_text
    return (image_text,)


@app.cell
def __(mo):
    harmful_threshold = mo.ui.slider(
        0, 1, debounce=True, label="Harmful Threshold", full_width=True, step=0.01
    )
    harmful_threshold
    return (harmful_threshold,)


@app.cell
def __(mo):
    run = mo.ui.run_button(full_width=True, label="Evaluate Meme")
    run
    return (run,)


@app.cell
def __(evaluate, harmful_threshold, image_file, image_text, mo, run):
    mo.stop(
        not run.value or not image_file or image_text.value == "",
        "Click `run` to submit the slider's value and make sure inputs are provided.",
    )

    probability, harmful = evaluate(
        image_file, image_text.value, harmful_threshold.value
    )
    return harmful, probability


@app.cell
def __(harmful, mo, probability):
    mo.md(
        """Image with probability of {:.2%} is considered {}.""".format(
            probability, "harmful" if harmful else "unharmful"
        )
    )
    return


if __name__ == "__main__":
    app.run()
