# Sound Effects Folder

Place your sound effect files here.

## Supported Formats
- MP3 (.mp3)
- WAV (.wav)
- AAC (.aac)
- OGG (.ogg)

## Example Structure
```
effects_sound/
├── whoosh.mp3
├── impact.wav
├── success.mp3
├── transition.wav
└── notification.mp3
```

## Usage in Config

Reference sounds by filename in your config:

```yaml
sound_effects:
  folder: effects_sound
  fadeout: true
  fadeout_duration: 1.0
  sounds:
    - name: whoosh.mp3
      time: 1.0        # Play at 1 second
      volume: 0.8      # 80% volume
    - name: impact.wav
      time: 4.0
      volume: 0.7
```

## Recommended Sound Effects for Instagram

- **Whoosh** - For text appearances
- **Impact/Hit** - For emphasis
- **Success/Ding** - For positive moments
- **Transition** - For scene changes
- **Bass drop** - For dramatic reveals
