from pydantic import BaseModel, Field

def load_tools(mcp_server):
    @mcp_server.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    class BioData(BaseModel):
        age: int = Field(description="Age")
        gender: str = Field(description="Gender")
        occupation: str = Field(description="Occupation")
        interests: list[str] = Field(description="Interests")

    _db = {
        "Alice": BioData(age=30, gender="Female", occupation="Software Engineer", interests=["Python", "Databricks", "Hiking"]),
        "Bob": BioData(age=28, gender="Male", occupation="Data Scientist", interests=["Machine Learning", "Statistics", "Chess"]),
    }

    @mcp_server.tool
    def return_biodata(name: str) -> dict:
        """Return biodata for a given name."""
        if name not in _db:
            raise ValueError(f"Unknown person: {name}")
        return _db[name].model_dump()
