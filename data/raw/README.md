# Put the team's real samples here

Use one folder per person. Use the same short name everywhere, in lowercase, for example `alice`.

```
data/raw/
  images/alice/alice_neutral.jpg
  images/alice/alice_smiling.jpg
  images/alice/alice_surprised.jpg
  audio/alice/alice_yes_approve.wav
  audio/alice/alice_confirm_transaction.wav
  demo/alice/alice_face.jpg                new photo, not used in training
  demo/alice/alice_voice.wav               new recording, not used in training
  unauthorized/unknown_1_neutral.jpg       a consenting non-member, used to tune rejection
  unauthorized/unknown_1_voice.wav
  unauthorized/unknown_2_neutral.jpg       a second non-member
  unauthorized/unknown_2_voice.wav
  demo/unauthorized/unknown_face.jpg       a third non-member, for the demo
  demo/unauthorized/unknown_voice.wav
```

Tips for good results:
- Photos: face the camera in good light, with one face in the frame. Use `.jpg` or `.png`.
- Audio: record in a quiet room. Use `.wav`, or install ffmpeg to use `.m4a`, `.mp3`, `.ogg` or `.opus`.
- Get consent from everyone whose face or voice is included, because the repository may be public.

As soon as `images/` contains a member folder, `python run_pipeline.py` uses this data and stops using the stand-ins.
