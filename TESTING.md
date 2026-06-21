# Testing

## Uploads

- Upload a `.pdf` document from the web app and confirm it still opens a ready chat workspace.
- Open the web file picker and confirm both `.pdf` and `.md` files are visible/selectable without changing OS file filters manually.
- Upload a `.md` document from the shared upload flow and confirm it indexes successfully and is available as a normal document.
- Upload an empty `.md` file and confirm the API returns a validation error instead of creating a document.
