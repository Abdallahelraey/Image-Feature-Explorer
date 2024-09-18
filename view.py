import base64
import httpx
import mesop as me


@me.stateclass
class State:
  file: me.UploadedFile
  prediction: str = ""


@me.page(
  security_policy=me.SecurityPolicy(
    allowed_iframe_parents=["https://google.github.io"]
  ),
  path="/uploader",
)
def app():
  state = me.state(State)
  with me.box(style=me.Style(padding=me.Padding.all(15), display="flex", justify_content="center", align_items="center", height="100vh")):
    with me.box(style=me.Style(text_align="center")):
      me.uploader(
        label="Upload Image",
        accepted_file_types=["image/png"],
        on_upload=handle_upload,
        type="flat",
        color="primary",
        style=me.Style(font_weight="bold"),
      )

      if state.file.size:
        with me.box(style=me.Style(margin=me.Margin.all(10))):
          me.text(f"File name: {state.file.name}")
          me.text(f"File size: {state.file.size}")
          me.text(f"File type: {state.file.mime_type}")

        with me.box(style=me.Style(margin=me.Margin.all(10))):
          me.image(src=_convert_contents_data_url(state.file))

        if state.prediction:
          with me.box(style=me.Style(margin=me.Margin.all(10))):
            me.text(f"Predicted Digit: {state.prediction}")


async def handle_upload(event: me.UploadEvent):
  state = me.state(State)
  state.file = event.file

  # Send the file to the FastAPI endpoint
  async with httpx.AsyncClient() as client:
    files = {'file': (state.file.name, state.file.getvalue(), state.file.mime_type)}
    response = await client.post("http://127.0.0.1:8000/predict", files=files)
    if response.status_code == 200:
      state.prediction = response.json().get("Predicted", "Prediction failed")
    else:
      state.prediction = "Prediction failed"


def _convert_contents_data_url(file: me.UploadedFile) -> str:
  return (
    f"data:{file.mime_type};base64,{base64.b64encode(file.getvalue()).decode()}"
  )