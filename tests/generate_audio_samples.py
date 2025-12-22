#!/usr/bin/env python3

"""
Generate audio samples from 30 utterances using gTTS and convert to 16-bit WAV format.
"""

import os
import subprocess
import numpy as np
from gtts import gTTS
from pydub import AudioSegment
import tempfile


def generate_audio_samples():
    # Define the 30 utterances
    utterances = {
        "english": [
            "Lily called me yesterday to discuss the project deadline",
            "Lucy is going to be late for the meeting this afternoon",
            "Emma just finished her presentation and received excellent feedback",
            "Kevin helped me fix the technical issue with the server",
            "Cindy will be joining our team next week as the new manager",
            "Tony is the best salesperson in our department this quarter",
            "Amy sent me the report that I needed for the client meeting",
            "David won the championship game with a perfect score",
            "Jessica recommended this restaurant for our dinner tonight",
            "Michael will be traveling to Shanghai for the conference next month"
        ],
        "chinese": [
            "张伟明昨天在会议上做了重要的发言",
            "李建国是我们公司的技术总监",
            "王小红刚刚完成了她的硕士论文",
            "陈志强将代表公司参加国际展览",
            "刘芳芳今天请了病假没有来上班",
            "杨光明是我们部门最资深的工程师",
            "赵文静负责管理整个项目团队",
            "黄丽华昨晚在音乐会上表演得非常出色",
            "周永康是这次活动的主要组织者",
            "吴天明刚刚被任命为新的部门经理"
        ],
        "mixed": [
            "张伟明告诉我Lily会在明天的会议上做报告",
            "李建国说Lucy负责这个项目的市场推广工作",
            "王小红和Emma一起完成了这个技术方案",
            "陈志强推荐Tony来负责客户关系管理",
            "刘芳芳提到Amy昨天给她发了重要的邮件",
            "杨光明说David的技术能力非常出色",
            "赵文静安排Jessica下周来公司进行培训",
            "黄丽华告诉我Michael下周会来北京出差",
            "周永康说Cindy的提案得到了董事会的批准",
            "吴天明安排Kevin负责这个国际合作项目"
        ]
    }

    # Create output directory
    output_dir = "/home/luigi/sherpa/test_audio"
    os.makedirs(output_dir, exist_ok=True)

    print("Generating audio samples from 30 utterances...")
    
    for category, texts in utterances.items():
        print(f"\nGenerating {category} utterances...")
        
        for i, text in enumerate(texts):
            # Create filename
            filename = f"{category}_{i+1:02d}.wav"
            filepath = os.path.join(output_dir, filename)
            
            print(f"  Generating: {filename}")
            
            try:
                # Create temporary file for gTTS
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
                    temp_mp3_path = temp_mp3.name
                
                # Use gTTS to generate speech
                # For Chinese text, we need to specify lang='zh'
                if category == 'chinese':
                    tts = gTTS(text, lang='zh')
                else:
                    tts = gTTS(text, lang='en')
                
                tts.save(temp_mp3_path)
                
                # Convert to WAV using pydub
                audio = AudioSegment.from_mp3(temp_mp3_path)
                
                # Export as 16-bit WAV
                audio.export(filepath, format="wav", parameters=["-bitexact"])
                
                # Clean up temporary file
                os.unlink(temp_mp3_path)
                
                print(f"    ✓ Saved: {filepath}")
                
            except Exception as e:
                print(f"    ✗ Error generating {filename}: {e}")
                # Clean up temporary file if it exists
                if 'temp_mp3_path' in locals() and os.path.exists(temp_mp3_path):
                    os.unlink(temp_mp3_path)
    
    print(f"\n✓ Audio samples generated in {output_dir}")
    print(f"  Total files: {len(os.listdir(output_dir))}")


if __name__ == "__main__":
    generate_audio_samples()