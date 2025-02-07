import os
from typing import List
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from pydantic import BaseModel
from dotenv import dotenv_values

from youtube_idea_generator_crew.tools.SearchYouTubeTool import (
    YoutubeVideoSearchAndDetailsTool,
)

llm = LLM(
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY"),
    model=os.getenv("OPENAI_MODEL_NAME"),  # 本次使用的模型
    temperature=0.7,  # 发散的程度
    # timeout=None,# 服务请求超时
    # max_retries=2,# 失败重试最大次数
)

# llm = LLM(
#     # base_url=os.getenv("GEMINI_API_URL"),
#     api_key=os.getenv("GEMINI_API_KEY"),
#     model=os.getenv("GEMINI_MODEL_NAME"),
#     temperature=0.7
# )

# llm = LLM(
#     model="groq/llama-3.3-70b-versatile",
#     temperature=0.7
# )


class ResearchItem(BaseModel):
    title: str
    url: str
    view_count: int


class VideoIdea(BaseModel):
    score: int
    video_title: str
    description: str
    video_id: str
    comment_id: str
    research: List[ResearchItem]


class VideoIdeasList(BaseModel):
    video_ideas: List[VideoIdea]


@CrewBase
class YoutubeIdeaGeneratorCrew:
    """YoutubeIdeaGeneratorCrew"""

    @agent
    def comment_filter_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["comment_filter_agent"],
            llm=llm,
            verbose=True
        )

    @agent
    def video_idea_generator_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["video_idea_generator_agent"], 
            llm=llm,
            verbose=True
        )

    @agent
    def research_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["research_agent"],
            llm=llm,
            tools=[YoutubeVideoSearchAndDetailsTool()],
            verbose=True,
        )

    @agent
    def scoring_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["scoring_agent"], 
            llm=llm,
            verbose=True
        )

    @task
    def filter_comments_task(self) -> Task:
        return Task(
            config=self.tasks_config["filter_comments_task"],
        )

    @task
    def generate_video_ideas_task(self) -> Task:
        return Task(
            config=self.tasks_config["generate_video_ideas_task"],
        )

    @task
    def research_video_ideas_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_video_ideas_task"],
        )

    @task
    def score_video_ideas_task(self) -> Task:
        return Task(
            config=self.tasks_config["score_video_ideas_task"],
            output_pydantic=VideoIdeasList,
        )

    @crew
    def crew(self) -> Crew:
        """Creates the YoutubeIdeaGenerator crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
