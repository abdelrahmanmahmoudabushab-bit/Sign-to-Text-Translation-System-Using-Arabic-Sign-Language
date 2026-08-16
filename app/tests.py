import json
from django.contrib.auth.models import User
from django.test import TestCase, Client
from django.urls import reverse
from app.models import TranslationClip, TranslationSession


class SignoViewTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.staff_user = User.objects.create_user(
            username='admin',
            email='admin@example.com',
            password='password123',
            is_staff=True
        )

    def test_index_view(self):
        response = self.client.get(reverse('index'))
        self.assertEqual(response.status_code, 200)
        self.assertIn('translation_history', self.client.session)
        self.assertEqual(self.client.session['translation_history'], [])

    def test_dashboard_view(self):
        response = self.client.get(reverse('dashboard'))
        self.assertEqual(response.status_code, 200)

    def test_history_view(self):
        response = self.client.get(reverse('history'))
        self.assertEqual(response.status_code, 200)

    def test_api_telemetry(self):
        response = self.client.get(reverse('api_telemetry'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'online')
        self.assertIn('system', data)
        self.assertIn('inference_metrics', data)

    def test_api_new_signs(self):
        response = self.client.get(reverse('api_new_signs'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'success')
        self.assertIn('count', data)

    def test_api_ollama_status(self):
        response = self.client.get(reverse('api_ollama_status'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data.get('status'), ['online', 'offline'])

    def test_upload_video_no_file(self):
        response = self.client.post(reverse('upload_video'))
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertEqual(data.get('status'), 'failed')

    def test_smooth_sentence_invalid_json(self):
        response = self.client.post(
            reverse('smooth_sentence'),
            data="not-a-json",
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_smooth_sentence_no_words(self):
        response = self.client.post(
            reverse('smooth_sentence'),
            data=json.dumps({'words': []}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_api_control_jetson_unauthorized(self):
        response = self.client.post(
            reverse('api_control_jetson'),
            data=json.dumps({'action': 'preload_vlm'}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 403)

    def test_api_control_jetson_authorized(self):
        self.client.login(username='admin', password='password123')
        response = self.client.post(
            reverse('api_control_jetson'),
            data=json.dumps({'action': 'invalid_action'}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 400)

    def test_api_sessions_pagination(self):
        response = self.client.get(reverse('api_sessions') + '?page=1&per_page=5')
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get('status'), 'success')
        self.assertEqual(data.get('per_page'), 5)

    def test_api_session_delete_unauthorized(self):
        response = self.client.delete(reverse('api_session_delete', kwargs={'session_id': 1}))
        self.assertEqual(response.status_code, 403)


class SignoModelTests(TestCase):
    def test_clip_and_session_creation(self):
        clip = TranslationClip.objects.create(
            gesture="مرحبا",
            dialect="Saudi Arabic Sign Language",
            confidence=0.92
        )
        self.assertIsNotNone(clip.pk)
        self.assertEqual(str(clip), f'[{clip.created_at:%Y-%m-%d %H:%M}] "مرحبا" (Saudi Arabic Sign Language)')

        session = TranslationSession.objects.create(
            arabic_sentence="مرحبا بكم",
            english_sentence="Welcome all",
            dialect="Saudi Arabic Sign Language"
        )
        session.clips.add(clip)
        self.assertEqual(session.clip_count, 1)
