
<template>
  <div class="search-view">
    <h2>Search Memories</h2>
    <form @submit.prevent="searchMemories">
      <div>
        <label for="query">Query:</label>
        <input id="query" type="text" v-model="query" required />
      </div>
      <button type="submit">Search</button>
    </form>
    <div v-if="results.length > 0">
      <h3>Results:</h3>
      <ul>
        <li v-for="(result, index) in results" :key="index">
          Distance: {{ result.distance }}, Memory ID: {{ result.index }}
        </li>
      </ul>
    </div>
    <p v-if="message">{{ message }}</p>
  </div>
</template>

<script>
import axios from "axios";

export default {
  data() {
    return {
      query: "",
      results: [],
      message: null,
    };
  },
  methods: {
    async searchMemories() {
      try {
        const response = await axios.get("http://127.0.0.1:8000/search/", {
          params: { query: this.query, top_k: 5 },
        });
        const { distances, indices } = response.data.results;
        this.results = distances.map((distance, i) => ({
          distance,
          index: indices[i],
        }));
        this.message = null;
      } catch (error) {
        this.message = error.response?.data?.detail || "Error searching memories.";
      }
    },
  },
};
</script>

<style>
.search-view {
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #ccc;
  border-radius: 5px;
}
input {
  width: 100%;
  margin-bottom: 10px;
}
button {
  padding: 10px 15px;
  background-color: #ffc107;
  color: white;
  border: none;
  border-radius: 3px;
}
button:hover {
  background-color: #e0a800;
}
ul {
  list-style: none;
  padding: 0;
}
li {
  margin-bottom: 10px;
}
</style>
