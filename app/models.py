"""
Database models for Signo translation history.

TranslationClip  — one record per video segment translated (single gesture).
TranslationSession — groups multiple clips into a full sentence build.

Clips older than 3 days are purged by the `purge_old_clips` management command.
"""

import os

from django.db import models
from django.utils import timezone


def _clip_upload_path(instance, filename):
    """Store each clip in a date-partitioned folder to keep media/ tidy."""
    date = timezone.now().strftime('%Y/%m/%d')
    return os.path.join('sessions', date, filename)


class TranslationClip(models.Model):
    """
    One database row per recorded video segment sent to the AI pipeline.
    The video file is stored at MEDIA_ROOT/sessions/<year>/<month>/<day>/.
    """
    video = models.FileField(
        upload_to=_clip_upload_path,
        help_text="Recorded sign gesture video clip."
    )
    gesture = models.CharField(
        max_length=200,
        blank=True,
        help_text="Arabic word/phrase detected by the model."
    )
    dialect = models.CharField(
        max_length=100,
        default='Saudi Arabic Sign Language',
        help_text="Regional sign dialect used during this clip."
    )
    confidence = models.FloatField(
        default=0.0,
        help_text="Model prediction confidence (0.0 – 1.0)."
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Translation Clip'
        verbose_name_plural = 'Translation Clips'

    def __str__(self):
        return f'[{self.created_at:%Y-%m-%d %H:%M}] "{self.gesture}" ({self.dialect})'

    def delete(self, *args, **kwargs):
        """Remove the video file from disk when the record is deleted."""
        if self.video and os.path.isfile(self.video.path):
            try:
                os.remove(self.video.path)
            except OSError:
                pass
        super().delete(*args, **kwargs)


class TranslationSession(models.Model):
    """
    A complete interaction session: one or more clips → a built Arabic sentence.
    Created when the user presses '✨ Build Sentence'.
    """
    clips = models.ManyToManyField(
        TranslationClip,
        blank=True,
        related_name='sessions',
        help_text="Individual gesture clips that formed this sentence."
    )
    arabic_sentence = models.TextField(
        blank=True,
        help_text="AI-reconstructed Arabic output sentence."
    )
    english_sentence = models.TextField(
        blank=True,
        help_text="AI-reconstructed English translation."
    )
    dialect = models.CharField(
        max_length=100,
        default='Saudi Arabic Sign Language',
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Translation Session'
        verbose_name_plural = 'Translation Sessions'

    def __str__(self):
        preview = (self.arabic_sentence or '—')[:50]
        return f'[{self.created_at:%Y-%m-%d %H:%M}] {preview}'

    @property
    def clip_count(self):
        return self.clips.count()
