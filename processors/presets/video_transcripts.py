"""
Transcribe the speech in videos
"""
from backend.lib.preset import ProcessorPreset
from common.lib.compatibility import Compatibility, is_executable
from processors.machine_learning.audio_to_text import AudioToText

from common.lib.helpers import UserInput, convert_to_int


class VideoTranscriber(ProcessorPreset):
    """
    Run processor pipeline to transcribe the audio track of videos
    """
    type = "preset-video-transcripts"  # job type ID
    category = "Combined processors"  # category. 'Combined processors' are always listed first in the UI.
    title = "Transcribe videos"  # title displayed in UI
    description = "Extracts the audio track of each video and converts the speech in it to text, with either " \
                  "OpenAI's Whisper or GPT models. Runs 'Extract audio from videos' followed by 'Audio to text'."
    extension = "ndjson"

    # Allow on video datasets when ffmpeg is available; the first step of the
    # pipeline (audio-extractor) needs it to demux the audio track
    compatibility = Compatibility(media_types={"video"}, type_prefixes={"video-downloader"}, required_settings={("video-downloader.ffmpeg_path", is_executable)})

    references = [
        "[OpenAI Whisper blog](https://openai.com/research/whisper)",
        "[Whisper paper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)",
        "[How to use prompts](https://platform.openai.com/docs/guides/speech-to-text/prompting)",
    ]

    @classmethod
    def get_options(cls, parent_dataset=None, config=None) -> dict:
        """
        Get processor options

        These mirror the subset of 'Audio to text' options that are meaningful
        without seeing the intermediate audio dataset; the amount limits are
        deliberately split per host so the administrator's cap on locally
        transcribed files still applies.

        :param parent_dataset DataSet:  An object representing the dataset that
            the processor would be or was run on. Can be used, in conjunction with
            config, to show some options only to privileged users.
        :param config ConfigManager|None config:  Configuration reader (context-aware)
        :return dict:   Options for this processor
        """
        local_whisper = (
            True
            if (
                config.get("dmi-service-manager.bc_whisper_enabled", False)
                and config.get("dmi-service-manager.ab_server_address", False)
            )
            else False
        )

        options = {
            "model_host": {
                "type": UserInput.OPTION_CHOICE,
                "default": "local" if local_whisper else "openai",
                "options": {"openai": "OpenAI API"} | ({"local": "Local (DMI Service Manager)"} if local_whisper else {}),
                "help": "Model type",
                "tooltip": "Local Whisper models require DMI Service Manager to be running and configured in settings."
            },
        }

        if local_whisper:
            options.update({
                "amount_local": {
                    "type": UserInput.OPTION_TEXT,
                    "requires": "model_host==local"
                },
                "local_model": {
                    "type": UserInput.OPTION_CHOICE,
                    "help": "Whisper model",
                    "default": "small",
                    "tooltip": "Larger sizes increase quality at expense of greatly increasing the amount of time to process. Try the Small model and increase as needed.",
                    "options": {
                        "small.en": "Small English",
                        "small": "Small Detect Language",
                        "medium.en": "Medium English",
                        "medium": "Medium Detect Language",
                        "large": "Large Detect Language"
                    },
                    "requires": "model_host==local"
                },
                "translate": {
                    "type": UserInput.OPTION_TOGGLE,
                    "help": "Translate transcriptions to English",
                    "default": False,
                    "requires": "model_host==local"
                },
            })

            # Update the local amount max and help from config, as 'Audio to text' does
            max_number_audio_files = int(config.get("dmi-service-manager.bd_whisper_num_files", 100))
            if max_number_audio_files == 0:  # Unlimited allowed
                options["amount_local"]["help"] = "Number of videos"
                options["amount_local"]["default"] = 100
                options["amount_local"]["min"] = 0
                options["amount_local"]["tooltip"] = "Use '0' to transcribe all videos (this can take a very long time)"
            else:
                options["amount_local"]["help"] = f"Number of videos (max {max_number_audio_files})"
                options["amount_local"]["default"] = min(max_number_audio_files, 10)
                options["amount_local"]["max"] = max_number_audio_files
                options["amount_local"]["min"] = 1

        options.update({
            "amount_external": {
                "type": UserInput.OPTION_TEXT,
                "default": 10,
                "min": 0,
                "help": "Number of videos to transcribe (0 will transcribe all)",
                "tooltip": "Use '0' to transcribe all videos. Note that the OpenAI API is a paid service and will "
                           "count towards your API credit.",
                "requires": "model_host==openai"
            },
            "openai_action": {
                "type": UserInput.OPTION_CHOICE,
                "help": "Action to perform with OpenAI API",
                "default": "transcribe",
                "options": {
                    "transcribe": "Transcribe",
                    "translate": "Translate",
                    "diarize": "Diarize (Speaker identification)"
                },
                "requires": "model_host==openai",
                "tooltip": "Transcription converts speech to text in the original language. Translation converts speech to English text (only available with Whisper V2). Diarization separates speakers and attempts to group them by speaker (only available with GPT-4o Diarization)."
            },
            "openai_transcribe_model": {
                "type": UserInput.OPTION_CHOICE,
                "help": "Model",
                "default": "gpt-4o-transcribe",
                "tooltip": "GPT-4o generally outperforms Whisper",
                "options": {
                    "gpt-4o-transcribe": "GPT-4o",
                    "gpt-4o-mini-transcribe": "GPT-4o mini",
                    "whisper-1": "Whisper V2"
                },
                "requires": "model_host==openai&&openai_action==transcribe"
            },
            "prompt": {
                "type": UserInput.OPTION_TEXT,
                "help": "Prompt (optional)",
                "default": "",
                "tooltip": "Prompts can aid the model in specific vocabulary detection or to add punctuation"
                           "and filler words."
            },
            "language": {
                "type": UserInput.OPTION_TEXT,
                "help": "Language of audio",
                "default": "",
                "tooltip": "Optional; can help performance and latency. Use ISO-693-1 format (e.g. 'en').",
                "requires": "model_host==openai"
            },
            "save_annotations": {
                "type": UserInput.OPTION_ANNOTATION,
                "label": "Audio transcription",
                "tooltip": "Add transcriptions to top dataset",
                "default": False
            }
        })

        # Check for 4CAT wide API key if using OpenAI models
        if not config.get("api.openai.api_key"):
            options["api_key"] = {
                "type": UserInput.OPTION_TEXT,
                "default": "",
                "help": "OpenAI API key",
                "tooltip": "Can be created on platform.openapi.com",
                "requires": "model_host==openai",
                "sensitive": True
            }

        return options

    def get_processor_pipeline(self):
        """
        This queues a series of post-processors to transcribe videos.

        First, the audio track is demuxed from each video with ffmpeg; then the
        resulting audio archive is transcribed, either by a locally hosted
        Whisper model or through the OpenAI API.
        """
        model_host = self.parameters.get("model_host", "openai")

        if model_host == "local":
            amount = convert_to_int(self.parameters.get("amount_local", 10), 10)

            # The preset supplies these parameters directly, bypassing UserInput
            # sanitisation, so re-apply the administrator's cap here
            max_local = int(self.config.get("dmi-service-manager.bd_whisper_num_files", 100))
            if max_local:
                amount = min(amount, max_local) if amount else max_local

            transcriber_parameters = {
                "model_host": "local",
                "amount_local": amount,
                "local_model": self.parameters.get("local_model", "small"),
                "translate": self.parameters.get("translate", False),
            }
        else:
            amount = convert_to_int(self.parameters.get("amount_external", 10), 10)
            transcriber_parameters = {
                "model_host": "openai",
                "amount_external": amount,
                "openai_action": self.parameters.get("openai_action", "transcribe"),
                "openai_transcribe_model": self.parameters.get("openai_transcribe_model", "gpt-4o-transcribe"),
                "language": self.parameters.get("language", ""),
                "api_key": self.parameters.get("api_key", ""),
            }
            self.dataset.delete_parameter("api_key")  # sensitive, delete as soon as possible

        transcriber_parameters["prompt"] = self.parameters.get("prompt", "")
        transcriber_parameters["save_annotations"] = self.parameters.get("save_annotations", False)

        pipeline = [
            # first, extract the audio track from each video
            {
                "type": "audio-extractor",
                "parameters": {
                    "amount": amount
                }
            },
            # then, convert the speech in the extracted audio to text
            {
                "type": "audio-to-text",
                "parameters": transcriber_parameters
            }
        ]

        return pipeline

    @staticmethod
    def map_item(item):
        """
        Map a transcription to a legible item

        This dataset is a copy of the final 'Audio to text' result file, so it
        needs that processor's mapping to be readable here as well. Without it
        4CAT cannot determine the NDJSON's columns, which also rules out CSV
        export and the Explorer.

        :param item:  Transcription as returned by 'Audio to text'
        :return MappedItem:  Mapped item
        """
        return AudioToText.map_item(item)
