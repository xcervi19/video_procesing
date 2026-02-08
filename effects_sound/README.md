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

Reference sounds by filename in your config (sound_effects also supported):

```yaml
background_sound_intro:
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

Automatic insertion (silence detection):

```yaml
automatic_intro_background_sound:
  enabled: true
  folder: effects_sound
  sound: whoosh.mp3
  volume: 0.8
  fadeout: true
  fadeout_duration: 1.0
  min_silence_duration: 1.0
  silence_threshold: 0.001
  analysis_fps: 100
  max_insertions: 1
  insert_at: start
  insert_offset: 0.0
```

## Recommended Sound Effects for Instagram

- **Whoosh** - For text appearances
- **Impact/Hit** - For emphasis
- **Success/Ding** - For positive moments
- **Transition** - For scene changes
- **Bass drop** - For dramatic reveals
