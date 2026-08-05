from django.contrib import admin
from django.utils.html import format_html

from app.models import TranslationClip, TranslationSession


@admin.register(TranslationClip)
class TranslationClipAdmin(admin.ModelAdmin):
    list_display = ('id', 'gesture', 'dialect', 'confidence_pct', 'video_preview', 'created_at')
    list_filter = ('dialect', 'created_at')
    search_fields = ('gesture',)
    readonly_fields = ('created_at', 'video_preview')
    ordering = ('-created_at',)

    @admin.display(description='Confidence')
    def confidence_pct(self, obj):
        pct = int(obj.confidence * 100)
        color = '#10b981' if pct >= 70 else '#f59e0b' if pct >= 40 else '#ef4444'
        return format_html(
            '<span style="color:{}; font-weight:700;">{} %</span>', color, pct
        )

    @admin.display(description='Preview')
    def video_preview(self, obj):
        if obj.video:
            return format_html(
                '<video src="{}" controls style="max-width:320px; border-radius:8px;"></video>',
                obj.video.url
            )
        return '—'


class ClipInline(admin.TabularInline):
    model = TranslationSession.clips.through
    extra = 0
    verbose_name = 'Clip'
    verbose_name_plural = 'Linked Clips'


@admin.register(TranslationSession)
class TranslationSessionAdmin(admin.ModelAdmin):
    list_display = ('id', 'arabic_preview', 'english_preview', 'dialect', 'clip_count', 'created_at')
    list_filter = ('dialect', 'created_at')
    search_fields = ('arabic_sentence', 'english_sentence')
    readonly_fields = ('created_at', 'clip_count')
    inlines = [ClipInline]
    ordering = ('-created_at',)

    @admin.display(description='Arabic')
    def arabic_preview(self, obj):
        return (obj.arabic_sentence or '—')[:60]

    @admin.display(description='English')
    def english_preview(self, obj):
        return (obj.english_sentence or '—')[:60]

    @admin.display(description='Clips')
    def clip_count(self, obj):
        return obj.clips.count()
