<script>
  import MinimalTable from "./MinimalTable.svelte";
  import SkeletonPulse from "./SkeletonPulse.svelte";

  let { columns = [], rows = 6, class: tableClass = "" } = $props();
  let rowIndexes = $derived.by(() => Array.from({ length: rows }, (_, idx) => idx));
</script>

<MinimalTable class={tableClass}>
  <thead>
    <tr>
      {#each columns as column, index (`${column}-${index}`)}
        <th>{column}</th>
      {/each}
    </tr>
  </thead>
  <tbody>
    {#each rowIndexes as rowIndex (`row-${rowIndex}`)}
      <tr>
        {#each columns as _, colIndex (`cell-${rowIndex}-${colIndex}`)}
          <td>
            <div class="skeleton-cell">
              <SkeletonPulse
                width={colIndex === columns.length - 1 ? "62%" : "100%"}
                height={16}
              />
            </div>
          </td>
        {/each}
      </tr>
    {/each}
  </tbody>
</MinimalTable>

<style>
  .skeleton-cell {
    display: flex;
    align-items: center;
    min-height: 16px;
  }
</style>
