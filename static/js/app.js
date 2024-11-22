
import Vue from 'vue';

// Dynamically import Vue components
const MemoryForm = () => import('./components/MemoryForm.vue');
const GraphView = () => import('./components/GraphView.vue');
const SearchView = () => import('./components/SearchView.vue');

new Vue({
  el: '#app',
  components: {
    MemoryForm,
    GraphView,
    SearchView,
  },
});
