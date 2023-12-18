from czitools import read_tools
from czitools import metadata_tools


# define the file location
filepath = r"data/w96_A1+A2.czi"

# read all metadata and check for the attachment images
md = metadata_tools.CziMetadata(filepath)
print(f"CZI Image has Label {md.attachments.has_label}")
print(f"CZI Image has SlidePreview {md.attachments.has_preview}")
print(f"CZI Image has Prescan {md.attachments.has_prescan}")

# get info about attachments only
attachments = metadata_tools.CziAttachments(filepath)

for k, v in attachments.__dict__.items():
    print(k, v)

# read individual images as array and optionally copy the image files as CZIs
attachment_types = ["SlidePreview", "Label", "Prescan"]

for at in attachment_types:
    if at == "SlidePreview" and attachments.has_preview:
        data, loc = read_tools.read_attachments(filepath, attachment_type=at, copy=True)
        print(f"Shape of {at} image: {data.shape}")
        print(f"Location of {at}: {loc}")

    if at == "Label" and attachments.has_label:
        data, loc = read_tools.read_attachments(filepath, attachment_type=at, copy=True)
        print(f"Shape of {at} image: {data.shape}")
        print(f"Location of {at}: {loc}")

    if at == "Prescan" and attachments.has_prescan:
        data, loc = read_tools.read_attachments(filepath, attachment_type=at, copy=True)
        print(f"Shape of {at} image: {data.shape}")
        print(f"Location of {at}: {loc}")
