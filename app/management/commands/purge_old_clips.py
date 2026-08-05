"""
Management command: purge_old_clips

Deletes TranslationClip records (and their video files) older than N days.
Also deletes orphaned TranslationSession records with no clips.

Default retention: 3 days.

Usage:
    python manage.py purge_old_clips
    python manage.py purge_old_clips --days 7

Add to cron on the Jetson for automatic cleanup:
    0 3 * * * cd /home/jetson/Sign-to-Text-Translation-System-Using-Arabic-Sign-Language && \
              .venv/bin/python manage.py purge_old_clips >> logs/purge.log 2>&1
"""

from django.core.management.base import BaseCommand
from django.utils import timezone
import datetime


class Command(BaseCommand):
    help = 'Delete video clips and sessions older than N days (default: 3)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days', type=int, default=3,
            help='Retention period in days (default: 3)'
        )
        parser.add_argument(
            '--dry-run', action='store_true',
            help='Show what would be deleted without actually deleting.'
        )

    def handle(self, *args, **options):
        from app.models import TranslationClip, TranslationSession

        days = options['days']
        dry_run = options['dry_run']
        cutoff = timezone.now() - datetime.timedelta(days=days)

        # Find old clips
        old_clips = TranslationClip.objects.filter(created_at__lt=cutoff)
        clip_count = old_clips.count()

        # Find orphaned sessions (no clips or all clips deleted)
        old_sessions = TranslationSession.objects.filter(created_at__lt=cutoff)
        session_count = old_sessions.count()

        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f'[DRY RUN] Would delete {clip_count} clips and {session_count} sessions '
                    f'older than {days} days (cutoff: {cutoff:%Y-%m-%d %H:%M}).'
                )
            )
            return

        # Delete clips (triggers file deletion via model.delete())
        deleted_clips = 0
        for clip in old_clips.iterator():
            clip.delete()
            deleted_clips += 1

        # Delete old sessions
        deleted_sessions, _ = old_sessions.delete()

        self.stdout.write(
            self.style.SUCCESS(
                f'Purged {deleted_clips} clips and {deleted_sessions} sessions '
                f'older than {days} days.'
            )
        )
